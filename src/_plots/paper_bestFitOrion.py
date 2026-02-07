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
    get_unique_ndens, parse_simulation_params, resolve_data_input, info_simulations
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
    """M42/EON observational constraints with multi-tracer support.

    MAIN SCIENCE TENSION: Order-of-magnitude discrepancy in shell mass estimates
    =============================================================================
    - [CII] 158 µm (SOFIA/upGREAT): M_shell ~ 10^3 M_sun
    - HI 21 cm (FAST+VLA):          M_shell ~ 10^2 M_sun

    This factor of ~10 discrepancy is the central question this analysis addresses.

    Physical interpretation:
    - The EON is a blister H II region, not a spherical shell
    - [CII] traces the back hemisphere (PDR interacting with OMC-1)
    - HI traces the front hemisphere (expanding into low-density ISM)
    - The two tracers sample fundamentally different parts of an asymmetric structure

    References:
    - Pabst et al. (2019, 2020): [CII] observations, combined mass ~2000 M_sun
    - HI observations: Shell mass ~100 M_sun in front hemisphere
    """
    # Expansion velocity (same for all tracers)
    v_obs: float = 13.0          # km/s - well-constrained
    v_err: float = 2.0           # km/s

    # ==========================================================================
    # SHELL MASS - THE KEY TENSION POINT
    # ==========================================================================
    # HI tracer: ~10^2 M_sun (front hemisphere, expanding toward us)
    M_shell_HI: float = 100.0        # M_sun
    M_shell_HI_err: float = 30.0     # M_sun

    # [CII] tracer: ~10^3 M_sun (back hemisphere, PDR at OMC interface)
    M_shell_CII: float = 1000.0      # M_sun
    M_shell_CII_err: float = 300.0   # M_sun

    # Combined estimate (spherical assumption - likely overestimate)
    M_shell_combined: float = 2000.0 # M_sun - Pabst+2019
    M_shell_combined_err: float = 500.0  # M_sun

    # Dynamical age
    t_obs: float = 0.2           # Myr
    t_err: float = 0.05          # Myr

    # Shell radius
    R_obs: float = 4.0           # pc
    R_err: float = 0.5           # pc

    # Stellar mass (derived constraint)
    Mstar_obs: float = 34.0      # M_sun (theta^1 Ori C dominated)
    Mstar_err: float = 5.0       # M_sun

    @property
    def mass_ratio_CII_HI(self) -> float:
        """The [CII]/HI mass ratio - quantifies the main tension."""
        return self.M_shell_CII / self.M_shell_HI

    @property
    def p_HI(self) -> float:
        """Momentum from HI mass [M_sun km/s]."""
        return self.M_shell_HI * self.v_obs

    @property
    def p_CII(self) -> float:
        """Momentum from [CII] mass [M_sun km/s]."""
        return self.M_shell_CII * self.v_obs

    @property
    def p_combined(self) -> float:
        """Momentum from combined mass [M_sun km/s]."""
        return self.M_shell_combined * self.v_obs


@dataclass
class AnalysisConfig:
    """Configuration for best-fit analysis."""
    # Visualization mode
    mode: Literal['2d', '3d'] = '2d'

    # Which observables to constrain (include in chi^2)
    # DEFAULT: Best fit based on v, t, R, M_star (NOT shell mass)
    constrain_v: bool = True
    constrain_M_shell: bool = False  # Shell mass NOT in chi^2 by default (use --include-mshell)
    constrain_t: bool = True
    constrain_R: bool = True
    constrain_Mstar: bool = True

    # Free parameter (not exposed in CLI, kept for internal compatibility)
    free_param: Optional[Literal['v', 'M_shell', 't', 'R']] = None

    # Filter by nCore (for 2D mode)
    nCore_filter: Optional[str] = None

    # Mass tracer selection (for plotting and optional chi^2)
    mass_tracer: Literal['HI', 'CII', 'combined', 'all'] = 'combined'

    # Blister geometry correction
    blister_mode: bool = False
    blister_fraction: float = 0.5  # Fraction of shell visible in blister geometry

    # Show all trajectories (instead of just top N)
    show_all: bool = False

    # Observational constraints
    obs: ObservationalConstraints = field(default_factory=ObservationalConstraints)

    def get_mass_constraint(self) -> Tuple[float, float]:
        """Return (M_obs, M_err) based on tracer selection."""
        if self.mass_tracer == 'HI':
            return self.obs.M_shell_HI, self.obs.M_shell_HI_err
        elif self.mass_tracer == 'CII':
            return self.obs.M_shell_CII, self.obs.M_shell_CII_err
        else:  # 'combined' or 'all'
            return self.obs.M_shell_combined, self.obs.M_shell_combined_err

    def get_constraint_string(self) -> str:
        """Build a string describing active constraints."""
        constraints = []
        if self.constrain_v and self.free_param != 'v':
            constraints.append(f"v={self.obs.v_obs:.0f}+/-{self.obs.v_err:.0f} km/s")
        if self.constrain_M_shell and self.free_param != 'M_shell':
            M_obs, M_err = self.get_mass_constraint()
            constraints.append(f"M_shell({self.mass_tracer})={M_obs:.0f}+/-{M_err:.0f} M_sun")
        if self.constrain_t and self.free_param != 't':
            constraints.append(f"t={self.obs.t_obs:.2f}+/-{self.obs.t_err:.2f} Myr")
        if self.constrain_R and self.free_param != 'R':
            constraints.append(f"R={self.obs.R_obs:.1f}+/-{self.obs.R_err:.1f} pc")
        if self.constrain_Mstar:
            constraints.append(f"M_star={self.obs.Mstar_obs:.0f}+/-{self.obs.Mstar_err:.0f} M_sun")
        if self.blister_mode:
            constraints.append(f"blister_frac={self.blister_fraction:.1f}")
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

    def get_filename_suffix(self) -> str:
        """Generate filename suffix based on mode, tracer, and free parameter."""
        suffix = ""
        if self.mode == '3d':
            suffix += "_3d"
        if self.mass_tracer != 'combined':
            suffix += f"_{self.mass_tracer}"
        if self.blister_mode:
            suffix += "_blister"
        if self.show_all:
            suffix += "_showall"
        if self.free_param:
            suffix += f"_estimate_{self.free_param}"
        return suffix

    def get_free_param_label(self) -> str:
        """Get display label for free parameter."""
        labels = {
            'v': 'v [km/s]',
            'M_shell': r'$M_{\rm shell}$ [$M_\odot$]',
            't': 't [Myr]',
            'R': 'R [pc]',
        }
        return labels.get(self.free_param, self.free_param)


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

    # Predicted time (when --no-t is used, this is the optimal time)
    t_predicted: Optional[float] = None
    t_predicted_range: Optional[Tuple[float, float]] = None  # 1-sigma range


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

    # Shell mass - use selected tracer constraint
    M_obs, M_err = config.get_mass_constraint()

    # Apply blister correction if enabled (TRINITY predicts full shell, observation sees fraction)
    M_shell_compare = sim_values['M_shell']
    if config.blister_mode:
        M_shell_compare = sim_values['M_shell'] * config.blister_fraction

    if np.isfinite(M_shell_compare) and M_err > 0:
        delta_M = (M_shell_compare - M_obs) / M_err
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


def get_shell_mass_for_comparison(M_shell: float, config: AnalysisConfig) -> float:
    """
    Apply blister correction to shell mass if enabled.

    TRINITY predicts the full shell mass, but in blister geometry
    we only observe a fraction. This function returns the mass value
    to compare against observations.

    Parameters
    ----------
    M_shell : float
        Raw shell mass from simulation [M_sun]
    config : AnalysisConfig
        Analysis configuration with blister settings

    Returns
    -------
    float
        Shell mass adjusted for blister geometry (if enabled)
    """
    if config.blister_mode:
        return M_shell * config.blister_fraction
    return M_shell


def compute_chi2_for_tracer(r, config: AnalysisConfig, M_obs: float, M_err: float,
                            include_mass: bool = None) -> float:
    """
    Canonical chi² computation for a specific mass tracer.

    This is THE single function to use for all chi² calculations involving
    tracer mass comparisons. It ensures consistent handling of:
    - Blister geometry correction
    - Config constraint flags
    - All chi² components (v, M, t, R, M_star)

    Parameters
    ----------
    r : SimulationResult
        Simulation result object
    config : AnalysisConfig
        Analysis configuration
    M_obs : float
        Observed shell mass for this tracer [M_sun]
    M_err : float
        Shell mass uncertainty [M_sun]
    include_mass : bool, optional
        If True, always include mass term in chi² (useful for tracer comparisons
        even when constrain_M_shell is False). If None, uses config.constrain_M_shell.

    Returns
    -------
    float
        Total chi² value
    """
    obs = config.obs
    chi2 = 0.0

    # Velocity
    if config.constrain_v and np.isfinite(r.v_kms):
        chi2 += ((r.v_kms - obs.v_obs) / obs.v_err) ** 2

    # Shell mass - with blister correction
    use_mass = include_mass if include_mass is not None else config.constrain_M_shell
    if use_mass and np.isfinite(r.M_shell):
        M_compare = get_shell_mass_for_comparison(r.M_shell, config)
        chi2 += ((M_compare - M_obs) / M_err) ** 2

    # Age
    if config.constrain_t and np.isfinite(r.t_actual):
        chi2 += ((r.t_actual - obs.t_obs) / obs.t_err) ** 2

    # Radius
    if config.constrain_R and np.isfinite(r.R2):
        chi2 += ((r.R2 - obs.R_obs) / obs.R_err) ** 2

    # Stellar mass
    if config.constrain_Mstar and np.isfinite(r.Mstar):
        chi2 += ((r.Mstar - obs.Mstar_obs) / obs.Mstar_err) ** 2

    return chi2


def get_n_params_for_grid(mode: str) -> int:
    """
    Get number of fitted parameters for confidence region thresholds.

    For parameter-space confidence contours (not goodness-of-fit), we use
    the number of parameters being explored, not the number of constraints.

    Parameters
    ----------
    mode : str
        '2d' for (mCloud, sfe) grids, '3d' for (mCloud, sfe, nCore) grids

    Returns
    -------
    int
        Number of parameters (2 or 3)
    """
    return 3 if mode == '3d' else 2


def nCore_matches(ndens_str: str, filter_str: str) -> bool:
    """
    Check if nCore value matches filter, handling numeric formatting differences.

    Compares numerically to handle cases like '1e4' vs '1e04' vs '10000'.

    Parameters
    ----------
    ndens_str : str
        nCore value from simulation (e.g., '1e04')
    filter_str : str
        Filter value from CLI (e.g., '1e4')

    Returns
    -------
    bool
        True if values match numerically
    """
    try:
        return float(ndens_str) == float(filter_str)
    except (ValueError, TypeError):
        # Fall back to string comparison if conversion fails
        return ndens_str == filter_str


def run_debug_checks():
    """
    Run self-tests to verify internal consistency of chi² calculations.

    Tests:
    - Blister correction application
    - Chi² component calculations
    - n_params for parameter-space thresholds
    - nCore filter numeric matching
    - Δχ² threshold values
    """
    print("=" * 60)
    print("Running debug checks...")
    print("=" * 60)

    passed = 0
    failed = 0

    # Create test config
    test_obs = ObservationalConstraints(
        v_obs=10.0, v_err=2.0,
        M_shell_HI=100.0, M_shell_HI_err=20.0,
        M_shell_CII=1000.0, M_shell_CII_err=200.0,
        t_obs=0.2, t_err=0.05,
        R_obs=2.0, R_err=0.3,
        Mstar_obs=34.0, Mstar_err=3.0,
    )

    # Test 1: Blister correction
    print("\n[Test 1] Blister correction...")
    config_no_blister = AnalysisConfig(obs=test_obs, blister_mode=False)
    config_blister = AnalysisConfig(obs=test_obs, blister_mode=True, blister_fraction=0.5)

    M_test = 200.0
    M_no_blister = get_shell_mass_for_comparison(M_test, config_no_blister)
    M_blister = get_shell_mass_for_comparison(M_test, config_blister)

    if M_no_blister == 200.0 and M_blister == 100.0:
        print(f"  PASS: no blister={M_no_blister}, blister(0.5)={M_blister}")
        passed += 1
    else:
        print(f"  FAIL: expected (200, 100), got ({M_no_blister}, {M_blister})")
        failed += 1

    # Test 2: n_params for grid
    print("\n[Test 2] n_params for grid...")
    n_2d = get_n_params_for_grid('2d')
    n_3d = get_n_params_for_grid('3d')

    if n_2d == 2 and n_3d == 3:
        print(f"  PASS: 2d={n_2d}, 3d={n_3d}")
        passed += 1
    else:
        print(f"  FAIL: expected (2, 3), got ({n_2d}, {n_3d})")
        failed += 1

    # Test 3: nCore filter matching
    print("\n[Test 3] nCore filter numeric matching...")
    tests = [
        ('1e4', '1e4', True),
        ('1e04', '1e4', True),
        ('10000', '1e4', True),
        ('1e4', '1e5', False),
        ('1.0e4', '1e4', True),
    ]
    test3_pass = True
    for ndens, filter_val, expected in tests:
        result = nCore_matches(ndens, filter_val)
        if result != expected:
            print(f"  FAIL: nCore_matches('{ndens}', '{filter_val}') = {result}, expected {expected}")
            test3_pass = False

    if test3_pass:
        print(f"  PASS: all {len(tests)} numeric matching tests passed")
        passed += 1
    else:
        failed += 1

    # Test 4: Δχ² thresholds
    print("\n[Test 4] Δχ² thresholds (2-param)...")
    thresholds = get_delta_chi2_thresholds(2)
    # For 2 DOF: 1σ ≈ 2.30, 2σ ≈ 6.18, 3σ ≈ 11.83
    if 2.2 < thresholds['1sigma'] < 2.4 and 6.1 < thresholds['2sigma'] < 6.3:
        print(f"  PASS: 1σ={thresholds['1sigma']:.2f}, 2σ={thresholds['2sigma']:.2f}, 3σ={thresholds['3sigma']:.2f}")
        passed += 1
    else:
        print(f"  FAIL: unexpected thresholds: {thresholds}")
        failed += 1

    # Test 5: Chi² calculation consistency
    print("\n[Test 5] Chi² calculation consistency...")

    # Create mock result
    class MockResult:
        def __init__(self):
            self.v_kms = 12.0
            self.M_shell = 120.0
            self.t_actual = 0.22
            self.R2 = 2.1
            self.Mstar = 35.0

    mock_r = MockResult()
    config_test = AnalysisConfig(
        obs=test_obs,
        constrain_v=True,
        constrain_M_shell=True,
        constrain_t=True,
        constrain_R=True,
        constrain_Mstar=True,
        blister_mode=False
    )

    chi2 = compute_chi2_for_tracer(mock_r, config_test, test_obs.M_shell_HI, test_obs.M_shell_HI_err)

    # Manual calculation
    chi2_v = ((12.0 - 10.0) / 2.0) ** 2  # 1.0
    chi2_M = ((120.0 - 100.0) / 20.0) ** 2  # 1.0
    chi2_t = ((0.22 - 0.2) / 0.05) ** 2  # 0.16
    chi2_R = ((2.1 - 2.0) / 0.3) ** 2  # 0.111
    chi2_Mstar = ((35.0 - 34.0) / 3.0) ** 2  # 0.111
    expected_chi2 = chi2_v + chi2_M + chi2_t + chi2_R + chi2_Mstar

    if abs(chi2 - expected_chi2) < 0.001:
        print(f"  PASS: chi²={chi2:.3f} (expected {expected_chi2:.3f})")
        passed += 1
    else:
        print(f"  FAIL: chi²={chi2:.3f} != expected {expected_chi2:.3f}")
        print(f"        Components: v={chi2_v:.3f}, M={chi2_M:.3f}, t={chi2_t:.3f}, R={chi2_R:.3f}, M*={chi2_Mstar:.3f}")
        failed += 1

    # Test 6: Chi² with blister
    print("\n[Test 6] Chi² with blister correction...")
    config_blister_test = AnalysisConfig(
        obs=test_obs,
        constrain_v=False,
        constrain_M_shell=True,
        constrain_t=False,
        constrain_R=False,
        constrain_Mstar=False,
        blister_mode=True,
        blister_fraction=0.5
    )

    # M_shell=120, with blister*0.5 = 60, obs=100±20 → chi²_M = ((60-100)/20)² = 4.0
    chi2_blister = compute_chi2_for_tracer(mock_r, config_blister_test, 100.0, 20.0)
    expected_blister = ((60.0 - 100.0) / 20.0) ** 2

    if abs(chi2_blister - expected_blister) < 0.001:
        print(f"  PASS: chi² with blister={chi2_blister:.3f} (expected {expected_blister:.3f})")
        passed += 1
    else:
        print(f"  FAIL: chi² with blister={chi2_blister:.3f} != expected {expected_blister:.3f}")
        failed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Debug checks: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


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

    # Get mass constraint for selected tracer
    M_obs, M_err = config.get_mass_constraint()

    if config.constrain_v and v_arr is not None:
        chi2_arr += ((v_arr - obs.v_obs) / obs.v_err) ** 2
    if config.constrain_M_shell and M_arr is not None:
        chi2_arr += ((M_arr - M_obs) / M_err) ** 2
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

        # Determine evaluation time: if --no-t, find optimal time; otherwise use t_obs
        t_predicted = None
        t_predicted_range = None

        if not config.constrain_t:
            # Find the time that minimizes chi² for unconstrained parameters
            time_result = find_matching_time(output, config)
            if time_result is not None:
                t_predicted = time_result['t_best']
                t_predicted_range = time_result['t_range_1sigma']
                eval_time = t_predicted
            else:
                eval_time = config.obs.t_obs  # Fallback
        else:
            eval_time = config.obs.t_obs

        # Get snapshot closest to evaluation time
        snap = output.get_at_time(eval_time, mode='closest', quiet=True)

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
            t_predicted=t_predicted,
            t_predicted_range=t_predicted_range,
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

        # Apply nCore filter if specified (numeric comparison handles 1e4 vs 1e04)
        if config.nCore_filter and not nCore_matches(ndens, config.nCore_filter):
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

    If mass_tracer='all', creates a 3-row figure with one row per tracer (HI, [CII], combined).

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

    nrows_grid, ncols_grid = len(mCloud_list), len(sfe_list)

    # Build lookup
    lookup = {(r.mCloud, r.sfe): r for r in data}

    # Build velocity and mass grids (same for all tracers)
    v_grid = np.full((nrows_grid, ncols_grid), np.nan)
    M_grid = np.full_like(v_grid, np.nan)

    for i, mCloud in enumerate(mCloud_list):
        for j, sfe in enumerate(sfe_list):
            if (mCloud, sfe) in lookup:
                r = lookup[(mCloud, sfe)]
                v_grid[i, j] = r.v_kms
                M_grid[i, j] = r.M_shell

    obs = config.obs
    # For parameter-space confidence regions, use n_params (not n_dof)
    n_params = get_n_params_for_grid(config.mode)
    thresholds = get_delta_chi2_thresholds(n_params)

    # Determine which tracers to plot
    if config.mass_tracer == 'all':
        tracer_configs = [
            ('HI 21cm', obs.M_shell_HI, obs.M_shell_HI_err, 'blue'),
            ('[CII] 158µm', obs.M_shell_CII, obs.M_shell_CII_err, 'darkorange'),
            ('Combined', obs.M_shell_combined, obs.M_shell_combined_err, 'red'),
        ]
        fig, axes = plt.subplots(3, 3, figsize=(15, 14), dpi=150)
    else:
        # Single tracer mode
        M_obs, M_err = config.get_mass_constraint()
        if config.mass_tracer == 'HI':
            tracer_name = 'HI 21cm'
            tracer_color = 'blue'
        elif config.mass_tracer == 'CII':
            tracer_name = '[CII] 158µm'
            tracer_color = 'darkorange'
        else:
            tracer_name = 'Combined'
            tracer_color = 'red'
        tracer_configs = [(tracer_name, M_obs, M_err, tracer_color)]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
        axes = axes.reshape(1, 3)  # Make 2D for consistent indexing

    for row_idx, (tracer_name, M_obs, M_err, tracer_color) in enumerate(tracer_configs):
        # Compute chi² grid for this tracer using canonical helper
        chi2_grid = np.full((nrows_grid, ncols_grid), np.nan)
        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                if (mCloud, sfe) in lookup:
                    r = lookup[(mCloud, sfe)]
                    chi2_grid[i, j] = compute_chi2_for_tracer(r, config, M_obs, M_err)

        # Find best fit for this tracer
        best_chi2 = np.inf
        best_i, best_j = 0, 0
        best_result = None
        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                if (mCloud, sfe) in lookup:
                    chi2_val = chi2_grid[i, j]
                    if np.isfinite(chi2_val) and chi2_val < best_chi2:
                        best_chi2 = chi2_val
                        best_i, best_j = i, j
                        best_result = lookup[(mCloud, sfe)]

        # --- Panel 1: Chi^2 heatmap ---
        ax1 = axes[row_idx, 0]
        chi2_min = np.nanmin(chi2_grid)
        chi2_max = np.nanmax(chi2_grid)

        cmap = plt.cm.viridis_r
        im1 = ax1.imshow(chi2_grid, cmap=cmap, aspect='auto',
                         norm=mcolors.LogNorm(vmin=max(0.1, chi2_min), vmax=min(1000, chi2_max)))

        cbar1 = plt.colorbar(im1, ax=ax1, label=r'$\chi^2_{\rm total}$')

        # Add confidence level lines to colorbar at chi2_min + Δχ² (not just Δχ²)
        for level_name, delta_chi2 in [('1sigma', thresholds['1sigma']),
                                        ('2sigma', thresholds['2sigma']),
                                        ('3sigma', thresholds['3sigma'])]:
            chi2_threshold = chi2_min + delta_chi2
            if chi2_min < chi2_threshold < chi2_max:
                cbar1.ax.axhline(y=chi2_threshold, color='k', linestyle='--', linewidth=0.8)

        # Mark best-fit cell with star in tracer color
        ax1.plot(best_j, best_i, marker='*', markersize=20, color=tracer_color,
                 markeredgecolor='k', markeredgewidth=1.5, zorder=10)

        # Add chi2 values as text
        for i in range(nrows_grid):
            for j in range(ncols_grid):
                chi2_val = chi2_grid[i, j]
                if np.isfinite(chi2_val):
                    text_color = 'white' if chi2_val > 10 else 'black'
                    ax1.text(j, i, f'{chi2_val:.1f}', ha='center', va='center',
                            fontsize=7, color=text_color, fontweight='bold')
                else:
                    ax1.text(j, i, 'X', ha='center', va='center',
                            fontsize=10, color='gray', alpha=0.5)

        ax1.set_xticks(range(ncols_grid))
        ax1.set_xticklabels([f'{int(s)/100:.2f}' for s in sfe_list], fontsize=8)
        ax1.set_yticks(range(nrows_grid))
        ax1.set_yticklabels([f'{float(m):.0e}' for m in mCloud_list], fontsize=8)
        ax1.set_xlabel(r'Star Formation Efficiency ($\epsilon$)')
        ax1.set_ylabel(r'Cloud Mass ($M_{\rm cloud}$ [$M_\odot$])')
        ax1.set_title(f'{tracer_name}: $\\chi^2$ Heatmap')

        # --- Panel 2: Velocity heatmap ---
        ax2 = axes[row_idx, 1]
        v_min = np.nanmin(v_grid)
        v_max = np.nanmax(v_grid)

        im2 = ax2.imshow(v_grid, cmap='coolwarm', aspect='auto',
                         vmin=min(v_min, obs.v_obs - 3*obs.v_err),
                         vmax=max(v_max, obs.v_obs + 3*obs.v_err))
        cbar2 = plt.colorbar(im2, ax=ax2, label='v [km/s]')

        # Add observed velocity lines
        cbar2.ax.axhline(y=obs.v_obs, color='k', linestyle='-', linewidth=2)
        cbar2.ax.axhline(y=obs.v_obs - obs.v_err, color='k', linestyle='--', linewidth=1)
        cbar2.ax.axhline(y=obs.v_obs + obs.v_err, color='k', linestyle='--', linewidth=1)

        # Add velocity values as text
        for i in range(nrows_grid):
            for j in range(ncols_grid):
                v_val = v_grid[i, j]
                if np.isfinite(v_val):
                    ax2.text(j, i, f'{v_val:.1f}', ha='center', va='center',
                            fontsize=7, color='black', fontweight='bold')

        ax2.set_xticks(range(ncols_grid))
        ax2.set_xticklabels([f'{int(s)/100:.2f}' for s in sfe_list], fontsize=8)
        ax2.set_yticks(range(nrows_grid))
        ax2.set_yticklabels([f'{float(m):.0e}' for m in mCloud_list], fontsize=8)
        ax2.set_xlabel(r'Star Formation Efficiency ($\epsilon$)')
        ax2.set_ylabel(r'Cloud Mass ($M_{\rm cloud}$ [$M_\odot$])')
        ax2.set_title(f'{tracer_name}: Velocity (obs: {obs.v_obs:.0f} km/s)')

        # Mark best-fit
        ax2.plot(best_j, best_i, marker='*', markersize=15, color=tracer_color,
                 markeredgecolor='k', markeredgewidth=1, zorder=10)

        # --- Panel 3: Shell mass heatmap ---
        ax3 = axes[row_idx, 2]
        M_min = np.nanmin(M_grid)
        M_max = np.nanmax(M_grid)

        im3 = ax3.imshow(M_grid, cmap='coolwarm', aspect='auto',
                         vmin=min(M_min, M_obs - 3*M_err),
                         vmax=max(M_max, M_obs + 3*M_err))
        cbar3 = plt.colorbar(im3, ax=ax3, label=r'$M_{\rm shell}$ [$M_\odot$]')

        # Add observed shell mass lines for this tracer
        cbar3.ax.axhline(y=M_obs, color=tracer_color, linestyle='-', linewidth=2)
        cbar3.ax.axhline(y=M_obs - M_err, color=tracer_color, linestyle='--', linewidth=1)
        cbar3.ax.axhline(y=M_obs + M_err, color=tracer_color, linestyle='--', linewidth=1)

        # Add mass values as text
        for i in range(nrows_grid):
            for j in range(ncols_grid):
                M_val = M_grid[i, j]
                if np.isfinite(M_val):
                    ax3.text(j, i, f'{M_val:.0f}', ha='center', va='center',
                            fontsize=7, color='black', fontweight='bold')

        ax3.set_xticks(range(ncols_grid))
        ax3.set_xticklabels([f'{int(s)/100:.2f}' for s in sfe_list], fontsize=8)
        ax3.set_yticks(range(nrows_grid))
        ax3.set_yticklabels([f'{float(m):.0e}' for m in mCloud_list], fontsize=8)
        ax3.set_xlabel(r'Star Formation Efficiency ($\epsilon$)')
        ax3.set_ylabel(r'Cloud Mass ($M_{\rm cloud}$ [$M_\odot$])')
        ax3.set_title(f'{tracer_name}: Shell Mass (obs: {M_obs:.0f} M$_\\odot$)')

        # Mark best-fit
        ax3.plot(best_j, best_i, marker='*', markersize=15, color=tracer_color,
                 markeredgecolor='k', markeredgewidth=1, zorder=10)

    # Build title
    if config.mass_tracer == 'all':
        title_lines = [f'M42 Best-Fit Analysis: $n_{{\\rm core}}$ = {nCore_value} cm$^{{-3}}$',
                       r'[CII] vs HI Shell Mass Tension — Comparing Best-Fits by Tracer']
    else:
        best_result = min(data, key=lambda x: x.chi2_total)
        title_lines = [f'M42 Best-Fit Analysis: $n_{{\\rm core}}$ = {nCore_value} cm$^{{-3}}$']
        title_lines.append(f'Best: mCloud={best_result.mCloud}, sfe={best_result.sfe_float:.2f}, '
                           f'$M_\\star$={best_result.Mstar:.1f} M$_\\odot$, $\\chi^2$={best_result.chi2_total:.2f}')
        if config.free_param and best_result.free_value is not None:
            title_lines.append(f'Predicted {config.get_free_param_label()}: {best_result.free_value:.2f}')

    fig.suptitle('\n'.join(title_lines), fontsize=12, y=1.02 if config.mass_tracer != 'all' else 0.995)

    plt.tight_layout()

    suffix = config.get_filename_suffix()
    out_pdf = output_dir / f'bestfit_n{nCore_value}_heatmap{suffix}.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


def plot_trajectory_comparison_2d(results: List[SimulationResult], config: AnalysisConfig,
                                   output_dir: Path, nCore_value: str, top_n: int = 5):
    """
    Overlay v(t), M_shell(t), and R(t) trajectories for a single nCore.

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
        Number of best-fit models to show (ignored if config.show_all is True)
    """
    # Filter for this nCore
    data = [r for r in results if r.nCore == nCore_value]

    if not data:
        return

    # Sort by chi2
    data_all_sorted = sorted(data, key=lambda x: x.chi2_total)

    # Decide which simulations to plot
    if config.show_all:
        data_to_plot = data_all_sorted
    else:
        data_to_plot = data_all_sorted[:top_n]

    # 3 subplots: velocity, mass, radius
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    ax_v, ax_m, ax_r = axes

    obs = config.obs

    # Color map for different simulations
    if config.show_all:
        # Use viridis colormap based on chi2 for show_all mode
        chi2_vals = [r.chi2_total for r in data_to_plot]
        chi2_min, chi2_max = min(chi2_vals), max(chi2_vals)
        norm = mcolors.LogNorm(vmin=max(0.1, chi2_min), vmax=max(1, chi2_max))
        cmap = plt.cm.viridis_r
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(data_to_plot)))

    for i, r in enumerate(data_to_plot):
        if r.t_full is None:
            continue

        t = r.t_full
        v = r.v_full_kms
        M = r.M_shell_full
        R = r.R_full

        if config.show_all:
            # Color by chi2, no individual labels
            color = cmap(norm(r.chi2_total))
            alpha = 0.4
            lw = 0.8
            label = None
        else:
            color = colors[i]
            alpha = 0.8
            lw = 1.5
            label = f"{r.mCloud}_sfe{r.sfe} (M$_\\star$={r.Mstar:.0f}, $\\chi^2$={r.chi2_total:.1f})"

        # Velocity trajectory
        if v is not None:
            ax_v.plot(t, v, color=color, lw=lw, label=label, alpha=alpha)

        # Mass trajectory
        if M is not None:
            ax_m.plot(t, M, color=color, lw=lw, label=label, alpha=alpha)

        # Radius trajectory
        if R is not None:
            ax_r.plot(t, R, color=color, lw=lw, label=label, alpha=alpha)

    # Highlight best-fit trajectory based on mass_tracer selection
    # This shows which model best matches each tracer's mass constraint
    if config.show_all and data_all_sorted:
        # Define tracer configurations: (name, M_obs, M_err, color, linestyle)
        tracer_highlight_configs = []
        if config.mass_tracer in ['HI', 'all']:
            tracer_highlight_configs.append(('HI', obs.M_shell_HI, obs.M_shell_HI_err, 'blue', '-'))
        if config.mass_tracer in ['CII', 'all']:
            tracer_highlight_configs.append(('[CII]', obs.M_shell_CII, obs.M_shell_CII_err, 'darkorange', '-'))
        if config.mass_tracer in ['combined', 'all']:
            tracer_highlight_configs.append(('Comb', obs.M_shell_combined, obs.M_shell_combined_err, 'red', '-'))

        # Find and highlight best-fit for each selected tracer
        for tracer_name, M_obs, M_err, color, ls in tracer_highlight_configs:
            # Find best model for this tracer using canonical helper
            best_for_tracer = min(data, key=lambda r: compute_chi2_for_tracer(r, config, M_obs, M_err))
            chi2_val = compute_chi2_for_tracer(best_for_tracer, config, M_obs, M_err)

            if best_for_tracer.t_full is not None:
                label = f"Best ({tracer_name}): {best_for_tracer.mCloud}_sfe{best_for_tracer.sfe} ($\\chi^2$={chi2_val:.1f})"
                if best_for_tracer.v_full_kms is not None:
                    ax_v.plot(best_for_tracer.t_full, best_for_tracer.v_full_kms,
                              color=color, lw=2.5, ls=ls, label=label, alpha=1.0, zorder=5)
                if best_for_tracer.M_shell_full is not None:
                    ax_m.plot(best_for_tracer.t_full, best_for_tracer.M_shell_full,
                              color=color, lw=2.5, ls=ls, label=label, alpha=1.0, zorder=5)
                if best_for_tracer.R_full is not None:
                    ax_r.plot(best_for_tracer.t_full, best_for_tracer.R_full,
                              color=color, lw=2.5, ls=ls, label=label, alpha=1.0, zorder=5)

    # Check if time is being predicted
    time_is_predicted = not config.constrain_t
    best_model = data_all_sorted[0] if data_all_sorted else None

    # --- Velocity panel ---
    # Time marker depends on whether t is constrained
    if time_is_predicted and best_model and best_model.t_predicted is not None:
        # Show predicted time as vertical line
        ax_v.axvline(best_model.t_actual, color='purple', linestyle='--', lw=2,
                     label=f't_pred = {best_model.t_actual:.3f} Myr', zorder=5)
        # Don't show observed time band, just observed v at best time
        ax_v.errorbar(best_model.t_actual, obs.v_obs, yerr=obs.v_err,
                      fmt='s', color='red', markersize=12, capsize=5, capthick=2,
                      label='Observed v', zorder=10, markeredgecolor='k')
    else:
        ax_v.errorbar(obs.t_obs, obs.v_obs, xerr=obs.t_err, yerr=obs.v_err,
                      fmt='s', color='red', markersize=12, capsize=5, capthick=2,
                      label='M42 Observed', zorder=10, markeredgecolor='k')
        ax_v.axvspan(obs.t_obs - obs.t_err, obs.t_obs + obs.t_err,
                     alpha=0.2, color='blue', zorder=1)
    ax_v.axhspan(obs.v_obs - obs.v_err, obs.v_obs + obs.v_err,
                 alpha=0.2, color='red', zorder=1)

    ax_v.set_xlabel('Time [Myr]')
    ax_v.set_ylabel('Shell Velocity [km/s]')
    ax_v.set_title('Velocity Evolution')
    ax_v.legend(loc='upper right', fontsize=7)
    ax_v.set_xlim(0, max(0.5, obs.t_obs * 2.5))
    # Use symlog scale to handle v<=0 values robustly
    ax_v.set_yscale('symlog', linthresh=1.0)
    ax_v.set_ylim(0.5, 100)
    ax_v.grid(True, alpha=0.3, which='both')

    # --- Mass panel (log scale) ---
    # Show [CII] vs HI tension bands
    tracer_bands = [
        (obs.M_shell_HI, obs.M_shell_HI_err, 'blue', r'HI ($\sim 10^2 M_\odot$)', 0.15),
        (obs.M_shell_CII, obs.M_shell_CII_err, 'darkorange', r'[CII] ($\sim 10^3 M_\odot$)', 0.15),
        (obs.M_shell_combined, obs.M_shell_combined_err, 'red', 'Combined', 0.08),
    ]

    for M_val, M_err, color, label, alpha in tracer_bands:
        ax_m.axhspan(M_val - M_err, M_val + M_err, alpha=alpha, color=color, zorder=1)
        # Time position depends on whether t is constrained
        if time_is_predicted and best_model and best_model.t_predicted is not None:
            ax_m.errorbar(best_model.t_actual, M_val, yerr=M_err,
                          fmt='s', color=color, markersize=10, capsize=4, capthick=1.5,
                          label=f'{label}', zorder=10,
                          markeredgecolor='k', markeredgewidth=0.5)
        else:
            ax_m.errorbar(obs.t_obs, M_val, xerr=obs.t_err, yerr=M_err,
                          fmt='s', color=color, markersize=10, capsize=4, capthick=1.5,
                          label=f'{label}', zorder=10,
                          markeredgecolor='k', markeredgewidth=0.5)

    if time_is_predicted and best_model and best_model.t_predicted is not None:
        ax_m.axvline(best_model.t_actual, color='purple', linestyle='--', lw=2, zorder=5)
    else:
        ax_m.axvspan(obs.t_obs - obs.t_err, obs.t_obs + obs.t_err,
                     alpha=0.1, color='gray', zorder=0)

    ax_m.set_xlabel('Time [Myr]')
    ax_m.set_ylabel(r'Shell Mass [$M_\odot$]')
    ax_m.set_title(r'Shell Mass ([CII]/HI $\times$' + f'{obs.mass_ratio_CII_HI:.0f})')
    ax_m.legend(loc='upper left', fontsize=7)
    ax_m.set_xlim(0, max(0.5, obs.t_obs * 2.5))
    ax_m.set_yscale('log')
    ax_m.set_ylim(10, 1e4)
    ax_m.grid(True, alpha=0.3, which='both')

    # --- Radius panel ---
    if time_is_predicted and best_model and best_model.t_predicted is not None:
        ax_r.errorbar(best_model.t_actual, obs.R_obs, yerr=obs.R_err,
                      fmt='s', color='green', markersize=12, capsize=5, capthick=2,
                      label=f'Observed R: {obs.R_obs}±{obs.R_err} pc', zorder=10, markeredgecolor='k')
        ax_r.axvline(best_model.t_actual, color='purple', linestyle='--', lw=2, zorder=5)
    else:
        ax_r.errorbar(obs.t_obs, obs.R_obs, xerr=obs.t_err, yerr=obs.R_err,
                      fmt='s', color='green', markersize=12, capsize=5, capthick=2,
                      label=f'M42 Observed: {obs.R_obs}±{obs.R_err} pc', zorder=10, markeredgecolor='k')
        ax_r.axvspan(obs.t_obs - obs.t_err, obs.t_obs + obs.t_err,
                     alpha=0.2, color='blue', zorder=1)
    ax_r.axhspan(obs.R_obs - obs.R_err, obs.R_obs + obs.R_err,
                 alpha=0.2, color='green', zorder=1)

    ax_r.set_xlabel('Time [Myr]')
    ax_r.set_ylabel('Shell Radius [pc]')
    ax_r.set_title('Radius Evolution')
    ax_r.legend(loc='upper left', fontsize=7)
    ax_r.set_xlim(0, max(0.5, obs.t_obs * 2.5))
    ax_r.set_ylim(0, None)
    ax_r.grid(True, alpha=0.3)

    # Build title with free parameter estimate
    best = data_all_sorted[0] if data_all_sorted else None
    title_lines = [f'M42 Trajectory Comparison: $n_{{\\rm core}}$ = {nCore_value} cm$^{{-3}}$']
    if config.show_all:
        title_lines.append(f'Showing all {len(data_to_plot)} simulations')
    if time_is_predicted and best and best.t_predicted is not None:
        title_lines.append(f'Predicted t = {best.t_actual:.3f} Myr (purple dashed line)')
    if config.free_param and best and best.free_value is not None:
        title_lines.append(f'Predicted {config.get_free_param_label()}: {best.free_value:.2f}')

    fig.suptitle('\n'.join(title_lines), fontsize=14, y=1.02)

    plt.tight_layout()

    suffix = config.get_filename_suffix()
    out_pdf = output_dir / f'bestfit_n{nCore_value}_trajectories{suffix}.pdf'
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

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    # Plot each simulation as a point
    v_vals = [r.v_kms for r in data if np.isfinite(r.v_kms)]
    M_vals = [r.M_shell for r in data if np.isfinite(r.M_shell)]
    chi2_vals = [r.chi2_total for r in data if np.isfinite(r.chi2_total)]

    if not v_vals:
        print(f"  No valid data points for residual plot (nCore={nCore_value})")
        return

    # Scatter plot colored by chi2
    scatter = ax.scatter(v_vals, M_vals, c=chi2_vals, cmap='viridis_r',
                         norm=mcolors.LogNorm(vmin=0.1, vmax=100),
                         s=100, edgecolors='k', linewidths=0.5, zorder=5)

    obs = config.obs
    # For parameter-space confidence regions, use n_params (not n_dof)
    n_params = get_n_params_for_grid(config.mode)
    thresholds = get_delta_chi2_thresholds(n_params)
    theta = np.linspace(0, 2 * np.pi, 100)

    # When mass_tracer='all', show contours for all three tracers
    if config.mass_tracer == 'all':
        tracer_obs_configs = [
            ('HI', obs.M_shell_HI, obs.M_shell_HI_err, 'blue'),
            ('[CII]', obs.M_shell_CII, obs.M_shell_CII_err, 'darkorange'),
            ('Comb', obs.M_shell_combined, obs.M_shell_combined_err, 'red'),
        ]

        # Draw 1-sigma ellipses for each tracer
        for tracer_name, M_obs, M_err, tracer_color in tracer_obs_configs:
            delta_chi2 = thresholds['1sigma']
            scale = np.sqrt(delta_chi2)
            v_ellipse = obs.v_obs + scale * obs.v_err * np.cos(theta)
            M_ellipse = M_obs + scale * M_err * np.sin(theta)
            ax.plot(v_ellipse, M_ellipse, color=tracer_color, lw=2, linestyle='--',
                    label=f'{tracer_name} 1$\\sigma$', alpha=0.8)

            # Mark observation point for this tracer
            ax.errorbar(obs.v_obs, M_obs, xerr=obs.v_err, yerr=M_err,
                        fmt='s', color=tracer_color, markersize=12, capsize=4, capthick=2,
                        zorder=10, markeredgecolor='k', markeredgewidth=1)

            # Mark best-fit for this tracer using canonical helper
            best_tracer = min(data, key=lambda r: compute_chi2_for_tracer(r, config, M_obs, M_err))
            chi2_val = compute_chi2_for_tracer(best_tracer, config, M_obs, M_err)
            if np.isfinite(best_tracer.v_kms) and np.isfinite(best_tracer.M_shell):
                ax.plot(best_tracer.v_kms, best_tracer.M_shell, marker='*', markersize=20,
                        color=tracer_color, markeredgecolor='k', markeredgewidth=1, zorder=15,
                        label=f'Best {tracer_name} ($\\chi^2$={chi2_val:.1f})')
    else:
        # Standard single-tracer: draw confidence ellipses
        M_obs_single, M_err_single = config.get_mass_constraint()
        for level_name, color, label in [('1sigma', 'green', r'1$\sigma$'),
                                          ('2sigma', 'orange', r'2$\sigma$'),
                                          ('3sigma', 'red', r'3$\sigma$')]:
            delta_chi2 = thresholds[level_name]
            scale = np.sqrt(delta_chi2)
            v_ellipse = obs.v_obs + scale * obs.v_err * np.cos(theta)
            M_ellipse = M_obs_single + scale * M_err_single * np.sin(theta)
            ax.plot(v_ellipse, M_ellipse, color=color, lw=2, linestyle='--',
                    label=f'{label} ($\\Delta\\chi^2={delta_chi2:.2f}$)')

        # Mark observation point
        ax.errorbar(obs.v_obs, M_obs_single,
                    xerr=obs.v_err, yerr=M_err_single,
                    fmt='s', color='red', markersize=15, capsize=5, capthick=2,
                    label='M42 Observed', zorder=10, markeredgecolor='k', markeredgewidth=2)

        # Mark best-fit
        best = min(data, key=lambda r: r.chi2_total)
        if np.isfinite(best.v_kms) and np.isfinite(best.M_shell):
            ax.plot(best.v_kms, best.M_shell, marker='*', markersize=25,
                    color='gold', markeredgecolor='k', markeredgewidth=1.5, zorder=15,
                    label=f'Best fit ($\\chi^2={best.chi2_total:.2f}$)')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label=r'$\chi^2$')

    # Labels
    ax.set_xlabel('Shell Velocity [km/s]')
    ax.set_ylabel(r'Shell Mass [$M_\odot$]')
    ax.set_yscale('log')
    ax.set_ylim(50, 5000)

    # Build title
    best = min(data, key=lambda r: r.chi2_total)
    title_lines = [f'M42 Parameter Space ($n_{{\\rm core}}$ = {nCore_value} cm$^{{-3}}$)',
                   f'at t = {obs.t_obs} Myr']
    if config.mass_tracer == 'all':
        title_lines.append('[CII] vs HI Mass Tension')
    if config.free_param and best.free_value is not None:
        title_lines.append(f'Predicted {config.get_free_param_label()}: {best.free_value:.2f}')

    ax.set_title('\n'.join(title_lines))
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    suffix = config.get_filename_suffix()
    out_pdf = output_dir / f'bestfit_n{nCore_value}_residuals{suffix}.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


def plot_Mstar_constraint_2d(results: List[SimulationResult], config: AnalysisConfig,
                              output_dir: Path, nCore_value: str):
    """
    Plot stellar mass constraint diagram showing M_star contours.

    If mass_tracer='all', creates a 1x3 panel showing chi² for each tracer
    (HI, [CII], combined) with M_star contours overlaid.

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
    nrows_grid, ncols_grid = len(mCloud_list), len(sfe_list)

    obs = config.obs
    lookup = {(r.mCloud, r.sfe): r for r in data}

    # Build Mstar grid (same for all tracers)
    j_coords, i_coords = np.meshgrid(range(ncols_grid), range(nrows_grid))
    Mstar_grid = np.full((nrows_grid, ncols_grid), np.nan)
    for i, mCloud in enumerate(mCloud_list):
        for j, sfe in enumerate(sfe_list):
            mCloud_f = float(mCloud)
            sfe_f = int(sfe) / 100.0
            Mstar_grid[i, j] = compute_stellar_mass(mCloud_f, sfe_f)

    Mstar_levels = [obs.Mstar_obs - 2*obs.Mstar_err,
                    obs.Mstar_obs - obs.Mstar_err,
                    obs.Mstar_obs,
                    obs.Mstar_obs + obs.Mstar_err,
                    obs.Mstar_obs + 2*obs.Mstar_err]

    # Determine which tracers to plot
    if config.mass_tracer == 'all':
        tracer_configs = [
            ('HI 21cm', obs.M_shell_HI, obs.M_shell_HI_err, 'blue'),
            ('[CII] 158µm', obs.M_shell_CII, obs.M_shell_CII_err, 'darkorange'),
            ('Combined', obs.M_shell_combined, obs.M_shell_combined_err, 'red'),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    else:
        # Single tracer mode
        M_obs, M_err = config.get_mass_constraint()
        if config.mass_tracer == 'HI':
            tracer_name = 'HI 21cm'
            tracer_color = 'blue'
        elif config.mass_tracer == 'CII':
            tracer_name = '[CII] 158µm'
            tracer_color = 'darkorange'
        else:
            tracer_name = 'Combined'
            tracer_color = 'red'
        tracer_configs = [(tracer_name, M_obs, M_err, tracer_color)]
        fig, axes = plt.subplots(1, 1, figsize=(10, 8), dpi=150)
        axes = [axes]  # Make iterable

    for ax_idx, (tracer_name, M_obs, M_err, tracer_color) in enumerate(tracer_configs):
        ax = axes[ax_idx]

        # Compute chi² grid for this tracer using canonical helper
        chi2_grid = np.full((nrows_grid, ncols_grid), np.nan)
        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                if (mCloud, sfe) in lookup:
                    r = lookup[(mCloud, sfe)]
                    chi2_grid[i, j] = compute_chi2_for_tracer(r, config, M_obs, M_err)

        # Plot heatmap
        im = ax.imshow(chi2_grid, cmap='viridis_r', aspect='auto',
                       norm=mcolors.LogNorm(vmin=0.1, vmax=100),
                       extent=[-0.5, ncols_grid-0.5, nrows_grid-0.5, -0.5])

        # Add M_star contours
        contour = ax.contour(j_coords, i_coords, Mstar_grid, levels=Mstar_levels,
                             colors=['gray', 'blue', 'black', 'blue', 'gray'],
                             linestyles=['--', '--', '-', '--', '--'],
                             linewidths=[1, 1.5, 2, 1.5, 1])
        ax.clabel(contour, inline=True, fontsize=8, fmt='M$_\\star$=%.0f')

        # Find best-fit for this tracer
        best_chi2 = np.inf
        best_i, best_j = 0, 0
        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                if (mCloud, sfe) in lookup:
                    chi2_val = chi2_grid[i, j]
                    if np.isfinite(chi2_val) and chi2_val < best_chi2:
                        best_chi2 = chi2_val
                        best_i, best_j = i, j

        # Mark best-fit with tracer-colored star
        ax.plot(best_j, best_i, marker='*', markersize=25, color=tracer_color,
                markeredgecolor='k', markeredgewidth=1.5, zorder=10)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label=r'$\chi^2_{\rm total}$')

        # Labels
        ax.set_xticks(range(ncols_grid))
        ax.set_xticklabels([f'{int(s)/100:.2f}' for s in sfe_list], fontsize=8)
        ax.set_yticks(range(nrows_grid))
        ax.set_yticklabels([f'{float(m):.0e}' for m in mCloud_list], fontsize=8)
        ax.set_xlabel(r'Star Formation Efficiency ($\epsilon$)')
        ax.set_ylabel(r'Cloud Mass ($M_{\rm cloud}$ [$M_\odot$])')

        # Title for this panel
        ax.set_title(f'{tracer_name}\n$M_{{\\rm obs}}$ = {M_obs:.0f} +/- {M_err:.0f} M$_\\odot$')

    # Build overall title
    if config.mass_tracer == 'all':
        fig.suptitle(f'M42 Stellar Mass Constraint: $n_{{\\rm core}}$ = {nCore_value} cm$^{{-3}}$\n'
                     r'[CII] vs HI Shell Mass Tension — M$_\star$(obs) = '
                     f'{obs.Mstar_obs:.0f} +/- {obs.Mstar_err:.0f} M$_\\odot$ (black line)',
                     fontsize=11, y=1.02)
    else:
        best = min(data, key=lambda r: r.chi2_total)
        title_lines = [f'M42 Stellar Mass Constraint: $n_{{\\rm core}}$ = {nCore_value} cm$^{{-3}}$',
                       f'M$_\\star$(obs) = {obs.Mstar_obs:.0f} +/- {obs.Mstar_err:.0f} M$_\\odot$ '
                       f'(black line = {obs.Mstar_obs:.0f} M$_\\odot$)']
        if config.free_param and best.free_value is not None:
            title_lines.append(f'Predicted {config.get_free_param_label()}: {best.free_value:.2f}')
        fig.suptitle('\n'.join(title_lines))

    plt.tight_layout()

    suffix = config.get_filename_suffix()
    out_pdf = output_dir / f'bestfit_n{nCore_value}_Mstar{suffix}.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


def plot_momentum_comparison(results: List[SimulationResult], config: AnalysisConfig,
                              output_dir: Path, nCore_value: str):
    """
    Compare TRINITY momentum predictions to [CII] and HI estimates.

    This addresses the [CII]/HI mass tension through momentum:
    - p_HI = M_HI × v ~ 10^2 × 13 ~ 1300 M_sun km/s
    - p_[CII] = M_[CII] × v ~ 10^3 × 13 ~ 13000 M_sun km/s

    Key question: Which momentum does TRINITY predict? This constrains
    whether the shell is better described by HI or [CII] mass.

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
    data = [r for r in results if r.nCore == nCore_value]
    if not data:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    obs = config.obs

    # --- Panel 1: p(t) trajectories ---
    ax1 = axes[0]

    if config.mass_tracer == 'all':
        # Plot best-fit trajectory for each tracer using canonical helper
        tracer_configs = [
            ('HI', obs.M_shell_HI, obs.M_shell_HI_err, 'blue'),
            ('[CII]', obs.M_shell_CII, obs.M_shell_CII_err, 'darkorange'),
            ('Comb', obs.M_shell_combined, obs.M_shell_combined_err, 'red'),
        ]
        for tracer_name, M_obs, M_err, tracer_color in tracer_configs:
            best_tracer = min(data, key=lambda r: compute_chi2_for_tracer(r, config, M_obs, M_err))
            if best_tracer.t_full is not None and best_tracer.v_full_kms is not None and best_tracer.M_shell_full is not None:
                p_full = best_tracer.M_shell_full * best_tracer.v_full_kms
                ax1.plot(best_tracer.t_full, p_full, color=tracer_color, lw=2.5,
                         label=f"Best({tracer_name})")
    else:
        # Default: plot top 5 by chi²
        data_sorted = sorted(data, key=lambda x: x.chi2_total)[:5]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data_sorted)))

        for i, r in enumerate(data_sorted):
            if r.t_full is None or r.v_full_kms is None or r.M_shell_full is None:
                continue
            p_full = r.M_shell_full * r.v_full_kms  # M_sun km/s
            ax1.plot(r.t_full, p_full, color=colors[i], lw=1.5,
                     label=f"{r.mCloud}_sfe{r.sfe}")

    # Mark observational momentum estimates - [CII] vs HI
    ax1.axhline(obs.p_HI, color='blue', ls='--', lw=2, label=f'p(HI) = {obs.p_HI:.0f}')
    ax1.axhline(obs.p_CII, color='darkorange', ls='--', lw=2, label=f'p([CII]) = {obs.p_CII:.0f}')
    ax1.axhline(2 * obs.p_HI, color='blue', ls=':', lw=1.5, label=f'2×p(HI) = {2*obs.p_HI:.0f}')
    ax1.axhline(obs.p_combined, color='red', ls='-.', lw=1.5, alpha=0.5, label=f'p(comb) = {obs.p_combined:.0f}')

    ax1.axvline(obs.t_obs, color='gray', ls=':', alpha=0.5)
    ax1.set_xlabel('Time [Myr]')
    ax1.set_ylabel(r'Momentum [$M_\odot$ km/s]')
    ax1.set_title('Momentum Evolution')
    ax1.legend(fontsize=7, loc='upper left')
    ax1.set_xlim(0, 0.5)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: M vs v with momentum contours ---
    ax2 = axes[1]

    v_vals = [r.v_kms for r in data if np.isfinite(r.v_kms)]
    M_vals = [r.M_shell for r in data if np.isfinite(r.M_shell)]
    chi2_vals = [r.chi2_total for r in data if np.isfinite(r.chi2_total)]

    if v_vals and M_vals:
        sc = ax2.scatter(v_vals, M_vals, c=chi2_vals, cmap='viridis_r',
                         norm=mcolors.LogNorm(vmin=0.1, vmax=100),
                         s=80, edgecolors='k', linewidths=0.5)

        # Momentum contours - [CII] vs HI
        v_grid = np.linspace(5, 25, 100)
        for p_val, ls, label in [(obs.p_HI, '--', 'p(HI)'),
                                  (obs.p_CII, '-', 'p([CII])'),
                                  (2*obs.p_HI, ':', '2×p(HI)')]:
            ax2.plot(v_grid, p_val / v_grid, 'k', ls=ls, alpha=0.3, lw=1)
            # Label at edge
            idx = len(v_grid) - 1
            if p_val / v_grid[idx] > 10:
                ax2.text(v_grid[idx], p_val / v_grid[idx], f' {label}', fontsize=7, alpha=0.6,
                         va='center')

        # Observational boxes - [CII] vs HI (the tension)
        ax2.errorbar(obs.v_obs, obs.M_shell_HI, xerr=obs.v_err, yerr=obs.M_shell_HI_err,
                     fmt='o', color='blue', markersize=10, capsize=4,
                     markeredgecolor='k', zorder=10, label=r'HI ($\sim 10^2 M_\odot$)')
        ax2.errorbar(obs.v_obs, obs.M_shell_CII, xerr=obs.v_err, yerr=obs.M_shell_CII_err,
                     fmt='^', color='darkorange', markersize=10, capsize=4,
                     markeredgecolor='k', zorder=10, label=r'[CII] ($\sim 10^3 M_\odot$)')
        ax2.errorbar(obs.v_obs, obs.M_shell_combined, xerr=obs.v_err, yerr=obs.M_shell_combined_err,
                     fmt='s', color='red', markersize=8, capsize=4,
                     markeredgecolor='k', zorder=10, alpha=0.5, label='Combined')

        # Mark best-fit point(s) with stars
        if config.mass_tracer == 'all':
            tracer_configs = [
                ('HI', obs.M_shell_HI, obs.M_shell_HI_err, 'blue'),
                ('[CII]', obs.M_shell_CII, obs.M_shell_CII_err, 'darkorange'),
                ('Comb', obs.M_shell_combined, obs.M_shell_combined_err, 'red'),
            ]
            for tracer_name, M_obs, M_err, tracer_color in tracer_configs:
                best_tracer = min(data, key=lambda r: compute_chi2_for_tracer(r, config, M_obs, M_err))
                ax2.plot(best_tracer.v_kms, best_tracer.M_shell, marker='*', markersize=20,
                         color=tracer_color, markeredgecolor='k', markeredgewidth=1, zorder=15)
        else:
            best = min(data, key=lambda r: r.chi2_total)
            ax2.plot(best.v_kms, best.M_shell, marker='*', markersize=20,
                     color='gold', markeredgecolor='k', markeredgewidth=1, zorder=15)

        ax2.set_xlabel('Velocity [km/s]')
        ax2.set_ylabel(r'Shell Mass [$M_\odot$]')
        ax2.set_title(f'M vs v at t={obs.t_obs} Myr')
        ax2.set_yscale('log')
        ax2.set_ylim(10, 5000)
        ax2.legend(fontsize=7, loc='upper right')
        plt.colorbar(sc, ax=ax2, label=r'$\chi^2$')

    # --- Panel 3: Momentum histogram ---
    ax3 = axes[2]

    p_at_t = [r.M_shell * r.v_kms for r in data
              if np.isfinite(r.M_shell) and np.isfinite(r.v_kms)]

    if p_at_t:
        ax3.hist(p_at_t, bins=15, color='gray', alpha=0.7, edgecolor='black')
        ax3.axvline(obs.p_HI, color='blue', ls='--', lw=2, label=f'p(HI) = {obs.p_HI:.0f}')
        ax3.axvline(obs.p_CII, color='darkorange', ls='--', lw=2, label=f'p([CII]) = {obs.p_CII:.0f}')
        ax3.axvline(2*obs.p_HI, color='blue', ls=':', lw=1.5, label=f'2×p(HI) = {2*obs.p_HI:.0f}')
        ax3.axvline(obs.p_combined, color='red', ls='-.', lw=1.5, alpha=0.5, label=f'p(comb)')

        # Mark best-fit momentum(s) using canonical helper
        if config.mass_tracer == 'all':
            # Show best-fit for each tracer
            tracer_configs = [
                ('HI', obs.M_shell_HI, obs.M_shell_HI_err, 'blue'),
                ('[CII]', obs.M_shell_CII, obs.M_shell_CII_err, 'darkorange'),
                ('Comb', obs.M_shell_combined, obs.M_shell_combined_err, 'red'),
            ]
            for tracer_name, M_obs, M_err, tracer_color in tracer_configs:
                best_tracer = min(data, key=lambda r: compute_chi2_for_tracer(r, config, M_obs, M_err))
                p_best = best_tracer.M_shell * best_tracer.v_kms
                ax3.axvline(p_best, color=tracer_color, ls='-', lw=3, alpha=0.8,
                           label=f'Best({tracer_name}): {p_best:.0f}')
        else:
            # Single tracer mode - show default best-fit
            best = min(data, key=lambda r: r.chi2_total)
            p_best = best.M_shell * best.v_kms
            ax3.axvline(p_best, color='gold', ls='-', lw=3, label=f'Best fit: {p_best:.0f}')

        ax3.set_xlabel(r'Momentum at t=0.2 Myr [$M_\odot$ km/s]')
        ax3.set_ylabel('Count')
        ax3.set_title('TRINITY Momentum Predictions')
        ax3.legend(fontsize=7, loc='upper right')

    fig.suptitle(r'[CII] vs HI Momentum Tension: $n_{\rm core}$ = ' + f'{nCore_value} cm' + r'$^{-3}$' + '\n'
                 r'p([CII]) $\sim$ ' + f'{obs.mass_ratio_CII_HI:.0f}' + r'$\times$ p(HI) — Which does TRINITY predict?',
                 fontsize=11)
    plt.tight_layout()

    suffix = config.get_filename_suffix()
    out_pdf = output_dir / f'bestfit_n{nCore_value}_momentum{suffix}.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


def plot_tracer_comparison(results: List[SimulationResult], config: AnalysisConfig,
                            output_dir: Path, nCore_value: str):
    """
    Side-by-side chi^2 heatmaps comparing [CII] vs HI mass constraints.

    This is the KEY FIGURE for the mass tension analysis:
    - Left panel: Best-fit using HI mass (~10^2 M_sun)
    - Middle panel: Best-fit using [CII] mass (~10^3 M_sun)
    - Right panel: Best-fit using combined mass (~2×10^3 M_sun)

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
    data = [r for r in results if r.nCore == nCore_value]
    if not data:
        return

    mCloud_list = sorted(set(r.mCloud for r in data), key=float)
    sfe_list = sorted(set(r.sfe for r in data), key=lambda x: int(x))
    nrows, ncols = len(mCloud_list), len(sfe_list)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    obs = config.obs

    # [CII] vs HI: the main tension comparison
    tracer_configs = [
        (r'HI 21cm ($\sim 10^2 M_\odot$)', obs.M_shell_HI, obs.M_shell_HI_err, axes[0], 'blue'),
        (r'[CII] 158$\mu$m ($\sim 10^3 M_\odot$)', obs.M_shell_CII, obs.M_shell_CII_err, axes[1], 'darkorange'),
        ('Combined (spherical)', obs.M_shell_combined, obs.M_shell_combined_err, axes[2], 'red'),
    ]

    # Shared colorbar normalization
    vmin, vmax = 0.1, 100

    for tracer_name, M_obs, M_err, ax, tracer_color in tracer_configs:
        # Recompute chi^2 with this tracer's mass
        chi2_grid = np.full((nrows, ncols), np.nan)
        lookup = {(r.mCloud, r.sfe): r for r in data}

        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                if (mCloud, sfe) in lookup:
                    r = lookup[(mCloud, sfe)]
                    # Recompute chi^2 with this mass constraint
                    chi2_v = ((r.v_kms - obs.v_obs) / obs.v_err)**2 if np.isfinite(r.v_kms) else 0
                    chi2_M = ((r.M_shell - M_obs) / M_err)**2 if np.isfinite(r.M_shell) else 0
                    chi2_t = ((r.t_actual - obs.t_obs) / obs.t_err)**2 if np.isfinite(r.t_actual) else 0
                    chi2_Mstar = ((r.Mstar - obs.Mstar_obs) / obs.Mstar_err)**2 if np.isfinite(r.Mstar) else 0
                    chi2_grid[i, j] = chi2_v + chi2_M + chi2_t + chi2_Mstar

        # Plot heatmap
        im = ax.imshow(chi2_grid, cmap='viridis_r', aspect='auto',
                       norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))

        # Mark best-fit
        if not np.all(np.isnan(chi2_grid)):
            best_idx = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
            ax.plot(best_idx[1], best_idx[0], marker='*', markersize=20, color='gold',
                    markeredgecolor='k', markeredgewidth=1.5, zorder=10)

            # Add chi^2 values
            for i in range(nrows):
                for j in range(ncols):
                    if np.isfinite(chi2_grid[i, j]):
                        color = 'white' if chi2_grid[i, j] > 10 else 'black'
                        ax.text(j, i, f'{chi2_grid[i, j]:.1f}', ha='center', va='center',
                                fontsize=7, color=color)

        ax.set_xticks(range(ncols))
        ax.set_xticklabels([f'{int(s)/100:.2f}' for s in sfe_list], fontsize=8)
        ax.set_yticks(range(nrows))
        ax.set_yticklabels([f'{float(m):.0e}' for m in mCloud_list], fontsize=8)
        ax.set_xlabel(r'SFE ($\epsilon$)')
        ax.set_ylabel(r'$M_{\rm cloud}$ [$M_\odot$]')
        ax.set_title(f'{tracer_name}', color=tracer_color, fontweight='bold', fontsize=10)

    # Shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
                                               cmap='viridis_r'),
                        cax=cbar_ax, label=r'$\chi^2_{\rm total}$')

    fig.suptitle(r'[CII] vs HI Mass Tension: $n_{\rm core}$ = ' + f'{nCore_value} cm' + r'$^{-3}$' + '\n'
                 r'Factor of $\sim$' + f'{obs.mass_ratio_CII_HI:.0f}' + r' discrepancy in shell mass estimates',
                 fontsize=11)

    suffix = config.get_filename_suffix()
    out_pdf = output_dir / f'bestfit_n{nCore_value}_tracer_comparison{suffix}.pdf'
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
    cmap = plt.cm.viridis_r

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

    # Build title with free parameter estimate
    title_lines = ['M42 Best-Fit Analysis (3D Mode)',
                   f'Global Best: mCloud={best.mCloud}, sfe={best.sfe_float:.2f}, '
                   f'nCore={best.nCore}, $\\chi^2$={best.chi2_total:.2f}']
    if config.free_param and best.free_value is not None:
        title_lines.append(f'Predicted {config.get_free_param_label()}: {best.free_value:.2f}')

    fig.suptitle('\n'.join(title_lines), fontsize=12, y=1.02)

    suffix = config.get_filename_suffix()
    out_pdf = output_dir / f'bestfit_faceted{suffix}.pdf'
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

    # Build title with free parameter estimate
    title_lines = ['M42 Parameter Space (3D)',
                   f'Best: mCloud={best.mCloud}, sfe={best.sfe_float:.2f}, '
                   f'nCore={best.nCore}, $\\chi^2$={best.chi2_total:.2f}']
    if config.free_param and best.free_value is not None:
        title_lines.append(f'Predicted {config.get_free_param_label()}: {best.free_value:.2f}')

    ax.set_title('\n'.join(title_lines), fontsize=12)

    ax.legend(loc='upper left')

    suffix = config.get_filename_suffix()
    out_pdf = output_dir / f'bestfit_scatter{suffix}.pdf'
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

    cmap = plt.cm.viridis_r
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

    # Build title with free parameter estimate
    best = min(results, key=lambda r: r.chi2_total)
    title_lines = ['M42 Marginal Projections (minimum $\\chi^2$ over marginalized parameter)']
    if config.free_param and best.free_value is not None:
        title_lines.append(f'Predicted {config.get_free_param_label()}: {best.free_value:.2f}')

    fig.suptitle('\n'.join(title_lines), fontsize=12, y=1.02)

    plt.tight_layout()

    suffix = config.get_filename_suffix()
    out_pdf = output_dir / f'bestfit_marginal_projections{suffix}.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


# =============================================================================
# Output Functions
# =============================================================================

def print_ranking_table(results: List[SimulationResult], config: AnalysisConfig,
                        top_n: int = 10):
    """Print ranked table of best-fit parameter combinations."""
    obs = config.obs

    sorted_results = sorted(results, key=lambda r: r.chi2_total)[:top_n]

    # Get chi2_min for Δχ² calculations
    chi2_min = sorted_results[0].chi2_total if sorted_results else 0.0

    # Get thresholds - use n_params=2 for parameter-space confidence
    n_params = get_n_params_for_grid(config.mode)
    thresholds = get_delta_chi2_thresholds(n_params)

    # Check if time is being predicted (--no-t mode)
    time_is_predicted = not config.constrain_t

    free_str = f"FREE: {config.free_param}" if config.free_param else "None"
    if time_is_predicted:
        free_str += " (t is PREDICTED, not constrained)"

    print("\n" + "=" * 120)
    print(f"TOP {top_n} BEST-FIT PARAMETER COMBINATIONS")
    print("=" * 120)
    print(f"Constraints: {config.get_constraint_string()}")
    if time_is_predicted:
        print("NOTE: Time (t) is not constrained - values evaluated at PREDICTED optimal time")
    print(f"Free parameter: {free_str}")
    print(f"chi²_min: {chi2_min:.2f}")
    print(f"Δχ² thresholds (n_params={n_params}): 1σ<{thresholds['1sigma']:.2f}, 2σ<{thresholds['2sigma']:.2f}, 3σ<{thresholds['3sigma']:.2f}")
    print("-" * 120)

    # Column header for time depends on whether it's predicted
    t_col_header = "t_pred" if time_is_predicted else "t_sim"

    if config.mass_tracer == 'all':
        # Show chi² for all three tracers (always include mass for tracer comparison)
        print(f"{'Rank':>4} {'mCloud':>10} {'sfe':>6} {'nCore':>8} {'M_star':>8} "
              f"{'v_sim':>7} {'M_sim':>8} {t_col_header:>7} {'chi2_HI':>8} {'chi2_CII':>9} {'chi2_comb':>10}")
        print(f"{'':>4} {'[M_sun]':>10} {'':>6} {'[cm-3]':>8} {'[M_sun]':>8} "
              f"{'[km/s]':>7} {'[M_sun]':>8} {'[Myr]':>7} {'':>8} {'':>9} {'':>10}")
        print("-" * 120)

        for i, r in enumerate(sorted_results, 1):
            # Use canonical chi² function with include_mass=True for tracer comparison
            chi2_HI = compute_chi2_for_tracer(r, config, obs.M_shell_HI, obs.M_shell_HI_err, include_mass=True)
            chi2_CII = compute_chi2_for_tracer(r, config, obs.M_shell_CII, obs.M_shell_CII_err, include_mass=True)
            chi2_comb = compute_chi2_for_tracer(r, config, obs.M_shell_combined, obs.M_shell_combined_err, include_mass=True)

            print(f"{i:>4} {r.mCloud:>10} {r.sfe_float:>6.2f} {r.nCore:>8} "
                  f"{r.Mstar:>8.1f} {r.v_kms:>7.1f} {r.M_shell:>8.0f} "
                  f"{r.t_actual:>7.3f} {chi2_HI:>8.2f} {chi2_CII:>9.2f} {chi2_comb:>10.2f}")
    else:
        # Standard single-tracer output with Δχ² significance
        print(f"{'Rank':>4} {'mCloud':>10} {'sfe':>6} {'nCore':>8} {'M_star':>8} "
              f"{'v_sim':>8} {'M_sim':>8} {t_col_header:>8} {'chi2':>8} {'Δχ²':>6} {'Sig':>5}")
        print(f"{'':>4} {'[M_sun]':>10} {'':>6} {'[cm-3]':>8} {'[M_sun]':>8} "
              f"{'[km/s]':>8} {'[M_sun]':>8} {'[Myr]':>8} {'':>8} {'':>6} {'':>5}")
        print("-" * 120)

        for i, r in enumerate(sorted_results, 1):
            # Significance uses Δχ² = χ² - χ²_min (NOT absolute χ²)
            delta_chi2 = r.chi2_total - chi2_min
            if delta_chi2 < thresholds['1sigma']:
                sig = "***"
            elif delta_chi2 < thresholds['2sigma']:
                sig = "**"
            elif delta_chi2 < thresholds['3sigma']:
                sig = "*"
            else:
                sig = ""

            print(f"{i:>4} {r.mCloud:>10} {r.sfe_float:>6.2f} {r.nCore:>8} "
                  f"{r.Mstar:>8.1f} {r.v_kms:>8.1f} {r.M_shell:>8.0f} "
                  f"{r.t_actual:>8.3f} {r.chi2_total:>8.2f} {delta_chi2:>6.2f} {sig:>5}")

    print("-" * 120)
    print(f"Legend: *** = 1σ (Δχ²<{thresholds['1sigma']:.2f}), "
          f"** = 2σ (Δχ²<{thresholds['2sigma']:.2f}), "
          f"* = 3σ (Δχ²<{thresholds['3sigma']:.2f})")
    print("=" * 120)

    # Best fit summary - show for each tracer when mass_tracer='all'
    if config.mass_tracer == 'all':
        print(f"\n** BEST FIT FOR EACH TRACER:")
        if time_is_predicted:
            print("   (Time is PREDICTED - values at optimal time for each model)")
        tracer_configs = [
            ('HI', obs.M_shell_HI, obs.M_shell_HI_err),
            ('[CII]', obs.M_shell_CII, obs.M_shell_CII_err),
            ('Combined', obs.M_shell_combined, obs.M_shell_combined_err),
        ]
        for tracer_name, M_obs, M_err in tracer_configs:
            best = min(results, key=lambda r: compute_chi2_for_tracer(r, config, M_obs, M_err))
            chi2_val = compute_chi2_for_tracer(best, config, M_obs, M_err)
            print(f"\n   {tracer_name} (M={M_obs:.0f}±{M_err:.0f} M_sun):")
            print(f"     Best: {best.mCloud} M_sun, sfe={best.sfe_float:.2f}, nCore={best.nCore}")
            print(f"     M_star={best.Mstar:.1f}, v={best.v_kms:.1f} km/s, M_shell={best.M_shell:.0f} M_sun")
            if time_is_predicted and best.t_predicted is not None:
                t_range_str = ""
                if best.t_predicted_range:
                    t_range_str = f" [1σ: {best.t_predicted_range[0]:.3f}-{best.t_predicted_range[1]:.3f}]"
                print(f"     t = {best.t_actual:.3f} Myr (PREDICTED){t_range_str}")
            print(f"     chi²={chi2_val:.2f}")
    else:
        best = sorted_results[0]
        M_obs_print, M_err_print = config.get_mass_constraint()
        print(f"\n** BEST FIT ({config.mass_tracer} tracer):")
        print(f"   mCloud = {best.mCloud} M_sun")
        print(f"   sfe = {best.sfe_float:.2f}")
        print(f"   nCore = {best.nCore} cm^-3")
        print(f"   M_star = {best.Mstar:.1f} M_sun (obs: {obs.Mstar_obs:.0f}+/-{obs.Mstar_err:.0f})")
        print(f"   v = {best.v_kms:.1f} km/s (obs: {obs.v_obs:.0f}+/-{obs.v_err:.0f})")
        print(f"   M_shell = {best.M_shell:.0f} M_sun (obs[{config.mass_tracer}]: {M_obs_print:.0f}+/-{M_err_print:.0f})")
        if time_is_predicted and best.t_predicted is not None:
            t_range_str = ""
            if best.t_predicted_range:
                t_range_str = f" [1σ: {best.t_predicted_range[0]:.3f}-{best.t_predicted_range[1]:.3f}]"
            print(f"   t = {best.t_actual:.3f} Myr (PREDICTED){t_range_str}")
        else:
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
    # Default output directory is {folder}/analysis/ per documentation
    output_dir = Path(output_dir) if output_dir else folder_path / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = AnalysisConfig()

    print(f"\nLoading sweep results from: {folder_path}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {config.mode.upper()}")
    print(f"Mass tracer: {config.mass_tracer}")
    if config.blister_mode:
        print(f"Blister geometry: ON (fraction={config.blister_fraction})")
    if config.show_all:
        print("Trajectory plots: showing ALL simulations")
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

    # Multi-tracer analysis plots (run regardless of mode)
    print("\n  Creating multi-tracer analysis plots...")
    if config.mode == '2d':
        if config.nCore_filter:
            nCore_list_multi = [config.nCore_filter]
        else:
            nCore_list_multi = sorted(set(r.nCore for r in results), key=float)
        for nCore in nCore_list_multi:
            plot_momentum_comparison(results, config, output_dir, nCore)
    else:
        # In 3D mode, run momentum comparison for each nCore
        nCore_list_multi = sorted(set(r.nCore for r in results), key=float)
        for nCore in nCore_list_multi:
            plot_momentum_comparison(results, config, output_dir, nCore)

    # Tracer comparison plot (shows all three mass tracers side-by-side)
    if config.mass_tracer == 'all' or config.mode == '2d':
        print("\n  Creating tracer comparison plots...")
        if config.mode == '2d':
            for nCore in nCore_list_multi:
                plot_tracer_comparison(results, config, output_dir, nCore)
        else:
            for nCore in nCore_list_multi:
                plot_tracer_comparison(results, config, output_dir, nCore)

    print(f"\n** All plots saved to: {output_dir}")


# =============================================================================
# Command-line Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="""TRINITY parameter sweep best-fit analysis for M42 (Orion Nebula)

MAIN SCIENCE QUESTION: [CII] vs HI Shell Mass Tension
=====================================================
Observations show an order-of-magnitude discrepancy in shell mass:
  - [CII] 158µm (SOFIA):  M_shell ~ 10^3 M_sun
  - HI 21cm (FAST+VLA):   M_shell ~ 10^2 M_sun

This script compares TRINITY simulations against both tracers to
constrain which mass estimate is more consistent with the models.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis - best fit based on chi^2(v, t, R, M_star)
  # Shell mass is shown in plots but NOT used for ranking
  python paper_bestFitOrion.py --folder sweep_orion/

  # Include shell mass in chi^2 (optional)
  python paper_bestFitOrion.py --folder sweep_orion/ --include-mshell

  # Include shell mass with specific tracer
  python paper_bestFitOrion.py --folder sweep_orion/ --include-mshell --mass-tracer CII

  # Compare all tracers in plots (mass tension visualization)
  python paper_bestFitOrion.py --folder sweep_orion/ --mass-tracer all

  # Show all simulation trajectories
  python paper_bestFitOrion.py --folder sweep_orion/ --showall

  # Filter by nCore value
  python paper_bestFitOrion.py --folder sweep_orion/ --nCore 1e4

  # 3D mode (full parameter space)
  python paper_bestFitOrion.py --folder sweep_orion/ --mode 3d

  # Without stellar mass constraint
  python paper_bestFitOrion.py --folder sweep_orion/ --no-Mstar

Default chi^2 calculation:
  chi^2 = chi^2_v + chi^2_t + chi^2_R + chi^2_Mstar
  (Shell mass NOT included unless --include-mshell is used)

Default Observational Constraints:
  v_expansion:     13 +/- 2 km/s       (Pabst et al. 2020)
  Age:             0.2 +/- 0.05 Myr    (Pabst et al. 2019, 2020)
  R_shell:         4 +/- 0.5 pc        (Pabst et al. 2019)
  M_star:          34 +/- 5 M_sun      (theta^1 Ori C)

Shell Mass (shown in plots, optional for chi^2):
  M_shell (HI):    100 +/- 30 M_sun    (front hemisphere)
  M_shell ([CII]): 1000 +/- 300 M_sun  (back hemisphere/PDR)
  M_shell (comb):  2000 +/- 500 M_sun  (Pabst et al. 2019)
        """
    )

    # Required (unless --debug-checks is used)
    parser.add_argument('--folder', '-F', default=None,
                        help='Path to sweep output folder (required unless --debug-checks)')

    # Output
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output directory (default: {folder}/analysis/)')

    # Mode selection
    parser.add_argument('--mode', '-m', choices=['2d', '3d'], default='2d',
                        help='Visualization mode: 2d (separate plots per nCore) or 3d (full space)')
    parser.add_argument('--nCore', '-n', default=None,
                        help='Filter by nCore value (e.g., "1e4"). Implies 2D mode.')
    parser.add_argument('--mCloud', nargs='+', default=None,
                        help='Filter simulations by cloud mass (e.g., --mCloud 1e6 1e7).')
    parser.add_argument('--sfe', nargs='+', default=None,
                        help='Filter simulations by SFE (e.g., --sfe 001 010).')
    parser.add_argument('--info', action='store_true',
                        help='Scan folder and print available mCloud, SFE, and nCore values.')

    # Constraint configuration
    # DEFAULT chi^2 = v + t + R + M_star (shell mass is NOT included by default)
    parser.add_argument('--include-mshell', action='store_true',
                        help='Include shell mass in chi^2 calculation (not included by default)')
    parser.add_argument('--no-Mstar', action='store_true',
                        help='Do not constrain stellar mass')
    parser.add_argument('--no-v', action='store_true',
                        help='Do not constrain velocity')
    parser.add_argument('--no-t', action='store_true',
                        help='Do not constrain age')
    parser.add_argument('--no-R', action='store_true',
                        help='Do not constrain shell radius')

    # Custom observational values
    parser.add_argument('--v-obs', type=float, default=13.0,
                        help='Observed velocity [km/s] (default: 13.0)')
    parser.add_argument('--v-err', type=float, default=2.0,
                        help='Velocity uncertainty [km/s] (default: 2.0)')
    parser.add_argument('--M-combined', type=float, default=2000.0,
                        help='Combined (spherical) shell mass [M_sun] (default: 2000.0)')
    parser.add_argument('--M-combined-err', type=float, default=500.0,
                        help='Combined shell mass uncertainty [M_sun] (default: 500.0)')
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

    # Multi-tracer mass selection (for plotting; also used in chi^2 if --include-mshell)
    parser.add_argument('--mass-tracer', choices=['HI', 'CII', 'combined', 'all'],
                        default='combined',
                        help='Mass tracer for plots (default: combined). Use with --include-mshell to constrain.')
    parser.add_argument('--blister', action='store_true',
                        help='Apply blister H II region geometry correction')
    parser.add_argument('--blister-fraction', type=float, default=0.5,
                        help='Fraction of shell visible in blister geometry (default: 0.5)')
    parser.add_argument('--M-HI', type=float, default=100.0,
                        help='HI-derived shell mass [M_sun] (default: 100.0)')
    parser.add_argument('--M-HI-err', type=float, default=30.0,
                        help='HI shell mass uncertainty [M_sun] (default: 30.0)')
    parser.add_argument('--M-CII', type=float, default=1000.0,
                        help='[CII]-derived shell mass [M_sun] ~10^3 (default: 1000.0)')
    parser.add_argument('--M-CII-err', type=float, default=300.0,
                        help='[CII] shell mass uncertainty [M_sun] (default: 300.0)')

    # Trajectory plot options
    parser.add_argument('--showall', action='store_true',
                        help='Show all simulation trajectories on v(t) and M(t) plots')

    # Debug/testing
    parser.add_argument('--debug-checks', action='store_true',
                        help='Run self-tests to verify internal consistency')

    args = parser.parse_args()

    # Run debug checks if requested
    if args.debug_checks:
        success = run_debug_checks()
        sys.exit(0 if success else 1)

    # Handle --info mode
    if args.info:
        if not args.folder:
            parser.error("--info requires --folder to be specified")
        info = info_simulations(args.folder)
        print("=" * 50)
        print(f"Simulation parameters in: {args.folder}")
        print("=" * 50)
        print(f"  Total simulations: {info['count']}")
        print(f"  mCloud values: {info['mCloud']}")
        print(f"  SFE values: {info['sfe']}")
        print(f"  nCore values: {info['ndens']}")
        sys.exit(0)

    # Require --folder for normal operation
    if not args.folder:
        parser.error("--folder is required (use --debug-checks to run self-tests)")

    # Build observational constraints with multi-tracer support
    obs = ObservationalConstraints(
        v_obs=args.v_obs, v_err=args.v_err,
        M_shell_HI=args.M_HI, M_shell_HI_err=args.M_HI_err,
        M_shell_CII=args.M_CII, M_shell_CII_err=args.M_CII_err,
        M_shell_combined=args.M_combined, M_shell_combined_err=args.M_combined_err,
        t_obs=args.t_obs, t_err=args.t_err,
        R_obs=args.R_obs, R_err=args.R_err,
        Mstar_obs=args.Mstar, Mstar_err=args.Mstar_err,
    )

    # If nCore specified, force 2D mode
    mode = '2d' if args.nCore else args.mode

    config = AnalysisConfig(
        mode=mode,
        constrain_v=not args.no_v,
        constrain_M_shell=args.include_mshell,  # Only include M_shell if explicitly requested
        constrain_t=not args.no_t,
        constrain_R=not args.no_R,
        constrain_Mstar=not args.no_Mstar,
        nCore_filter=args.nCore,
        mass_tracer=args.mass_tracer,
        blister_mode=args.blister,
        blister_fraction=args.blister_fraction,
        show_all=args.showall,
        obs=obs,
    )

    main(args.folder, args.output_dir, config)
