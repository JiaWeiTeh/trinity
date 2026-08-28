# -*- coding: utf-8 -*-
"""
Shared data/χ² layer for the single-object trajectory-fitting figure scripts.

Originally lived inside ``scratch/paper_ODIN.py`` (Orion), which
``paper/rosette/figures/paper_Rosette.py`` then imported.  That made the Rosette
figures depend on a local-only ``scratch/`` file; hoisting the machinery here
lets each object folder (``paper/orionNeAtHood/``, ``paper/rosette/``) import a
tracked, shared module and stay independent of the others.

Provides:
- ``ObservationalConstraints`` / ``AnalysisConfig`` / ``SimulationResult`` — the
  per-object observables, which of them enter χ², and one run's results.
- ``load_sweep_results`` — walk a sweep folder, evaluate each run at ``t_obs``,
  return results sorted by χ².
- ``smooth_trajectory``, ``compute_stellar_mass``, ``compute_chi2``,
  ``nCore_matches``, ``_trim_after_end`` — helpers used by the figure scripts.

The figure scripts own all plotting; nothing here touches matplotlib.

@author: Jia Wei Teh
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Tuple
from scipy.signal import savgol_filter

from paper._lib.grid_template import filter_sim_files_by_phii
from trinity._output.trinity_reader import (
    load_output, find_all_simulations, parse_simulation_params,
)
from trinity._functions.unit_conversions import INV_CONV

# Unit conversion: 1 km/s ~ 1.0227 pc/Myr
PC_MYR_TO_KM_S = 1.0 / 1.0227

# Global font size shared by this figure family (labels, ticks, legends).
FONTSIZE = 16


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class ObservationalConstraints:
    """Observational constraints for comparison."""
    # Expansion velocity
    v_obs: float = 13.0          # km/s
    v_err: float = 2.0           # km/s

    # Shell mass options
    M_shell_HI: float = 100.0        # M_sun
    M_shell_HI_err: float = 10.0     # M_sun
    M_shell_CII: float = 1100.0      # M_sun
    M_shell_CII_err: float = 100.0   # M_sun
    M_shell_combined: float = 2000.0 # M_sun
    M_shell_combined_err: float = 500.0  # M_sun

    # Dynamical age
    t_obs: float = 0.25          # Myr
    t_err: float = 0.05          # Myr

    # Shell radius (HI)
    R_obs: float = 1.8           # pc
    R_err: float = 0.2           # pc

    # Shell radius ([CII])
    R_obs_Pabst: float = 2.7     # pc
    R_err_Pabst: float = 0     # pc

    # Stellar mass (derived constraint)
    Mstar_obs: float = 34.0      # M_sun
    Mstar_err: float = 5.0       # M_sun

    @property
    def mass_ratio_CII_HI(self) -> float:
        """The [CII]/HI mass ratio."""
        return self.M_shell_CII / self.M_shell_HI


@dataclass
class AnalysisConfig:
    """Configuration for trajectory analysis."""
    # Which observables to constrain (include in chi^2)
    constrain_v: bool = True
    constrain_M_shell: bool = False
    constrain_t: bool = True
    constrain_R: bool = True
    constrain_Mstar: bool = True

    # Filter by nCore
    nCore_filter: Optional[str] = None

    # Mass tracer selection
    mass_tracer: Literal['HI', 'CII', 'combined', 'all'] = 'combined'

    # Show all trajectories
    show_all: bool = False

    # Combine multiple nCore values on same plot
    combine_nCore: bool = False

    # PHII suffix variant ("yes" keeps yesPHII + untagged; "no" keeps only noPHII)
    phii_mode: str = "yes"

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

    def get_filename_suffix(self) -> str:
        """Generate filename suffix based on configuration."""
        suffix = ""
        if self.mass_tracer != 'combined':
            suffix += f"_{self.mass_tracer}"
        if self.show_all:
            suffix += "_showall"
        if self.combine_nCore:
            suffix += "_combined"
        if self.phii_mode == "no":
            suffix += "_noPHII"
        return suffix


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
    phase_full: Optional[np.ndarray] = None
    rcloud: float = np.nan


# =============================================================================
# Core Functions
# =============================================================================

def smooth_trajectory(t, y, window_frac=0.05, polyorder=3):
    """
    Smooth a trajectory array to remove numerical jitters.

    Uses Savitzky-Golay filter which preserves the overall shape
    while smoothing out local discontinuities.

    Parameters
    ----------
    t : array-like
        Time array
    y : array-like
        Data array to smooth
    window_frac : float
        Fraction of data length to use as window size (default 0.05 = 5%)
    polyorder : int
        Polynomial order for the filter (default 3)

    Returns
    -------
    y_smooth : array
        Smoothed data array
    """
    if y is None or len(y) < 10:
        return y

    # Calculate window length (must be odd)
    window_length = int(len(y) * window_frac)
    if window_length < polyorder + 2:
        window_length = polyorder + 2
    if window_length % 2 == 0:
        window_length += 1

    # Apply Savitzky-Golay filter
    try:
        y_smooth = savgol_filter(y, window_length, polyorder)
        return y_smooth
    except Exception:
        return y


def compute_stellar_mass(mCloud, sfe):
    """
    Compute stellar mass from cloud parameters.

    M_star = sfe * mCloud / (1 - sfe)
    """
    mCloud = np.asarray(mCloud)
    sfe = np.asarray(sfe)

    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(sfe >= 1.0, np.inf, sfe * mCloud / (1.0 - sfe))

    if result.ndim == 0:
        return float(result)
    return result


def compute_chi2(sim_values: dict, config: AnalysisConfig) -> dict:
    """
    Compute chi^2 with configurable constraints.
    """
    obs = config.obs
    chi2_terms = {}
    residuals = {}

    # Velocity
    if np.isfinite(sim_values['v_kms']) and obs.v_err > 0:
        delta_v = (sim_values['v_kms'] - obs.v_obs) / obs.v_err
        chi2_terms['v'] = delta_v**2 if config.constrain_v else 0.0
        residuals['v'] = delta_v
    else:
        chi2_terms['v'] = np.inf if config.constrain_v else 0.0
        residuals['v'] = np.nan

    # Shell mass
    M_obs, M_err = config.get_mass_constraint()
    if np.isfinite(sim_values['M_shell']) and M_err > 0:
        delta_M = (sim_values['M_shell'] - M_obs) / M_err
        chi2_terms['M'] = delta_M**2 if config.constrain_M_shell else 0.0
        residuals['M'] = delta_M
    else:
        chi2_terms['M'] = np.inf if config.constrain_M_shell else 0.0
        residuals['M'] = np.nan

    # Time
    if np.isfinite(sim_values['t_actual']) and obs.t_err > 0:
        delta_t = (sim_values['t_actual'] - obs.t_obs) / obs.t_err
        chi2_terms['t'] = delta_t**2 if config.constrain_t else 0.0
        residuals['t'] = delta_t
    else:
        chi2_terms['t'] = np.inf if config.constrain_t else 0.0
        residuals['t'] = np.nan

    # Radius
    if np.isfinite(sim_values['R2']) and obs.R_err > 0:
        delta_R = (sim_values['R2'] - obs.R_obs) / obs.R_err
        chi2_terms['R'] = delta_R**2 if config.constrain_R else 0.0
        residuals['R'] = delta_R
    else:
        chi2_terms['R'] = np.inf if config.constrain_R else 0.0
        residuals['R'] = np.nan

    # Stellar mass constraint
    Mstar = compute_stellar_mass(sim_values['mCloud'], sim_values['sfe'])
    if np.isfinite(Mstar) and obs.Mstar_err > 0:
        delta_Mstar = (Mstar - obs.Mstar_obs) / obs.Mstar_err
        chi2_terms['Mstar'] = delta_Mstar**2 if config.constrain_Mstar else 0.0
        residuals['Mstar'] = delta_Mstar
    else:
        chi2_terms['Mstar'] = np.inf if config.constrain_Mstar else 0.0
        residuals['Mstar'] = np.nan

    chi2_total = sum(chi2_terms.values())

    return {
        'chi2_total': chi2_total,
        'chi2_v': chi2_terms['v'],
        'chi2_M': chi2_terms['M'],
        'chi2_t': chi2_terms['t'],
        'chi2_R': chi2_terms['R'],
        'chi2_Mstar': chi2_terms['Mstar'],
        'delta_v': residuals['v'],
        'delta_M': residuals['M'],
        'delta_t': residuals['t'],
        'delta_R': residuals['R'],
        'delta_Mstar': residuals['Mstar'],
        'Mstar': Mstar,
        'free_value': None,
    }


def nCore_matches(ndens_str: str, filter_str: str) -> bool:
    """Check if nCore value matches filter (handles 1e4 vs 1e04)."""
    try:
        return abs(float(ndens_str) - float(filter_str)) < 1e-6 * max(float(ndens_str), float(filter_str))
    except ValueError:
        return ndens_str == filter_str


def _trim_after_end(output, t, v2, M_shell, R):
    """Truncate time series at the first snapshot where the shell
    collapses (``isCollapse``) or dissolves (``isDissolved``).

    This prevents misleading constant/frozen tails from appearing on
    trajectory plots after the bubble has already ended.
    """
    if t is None or len(t) == 0:
        return t, v2, M_shell, R

    # Try the explicit boolean flags first
    try:
        is_collapse = np.asarray(output.get('isCollapse'), dtype=bool)
        is_dissolved = np.asarray(output.get('isDissolved'), dtype=bool)
        ended = is_collapse | is_dissolved
    except (TypeError, ValueError):
        ended = None

    # Find the first index where the simulation has ended
    cut = None
    if ended is not None and np.any(ended):
        cut = int(np.argmax(ended))   # first True

    if cut is not None and cut > 0:
        # Keep one extra point at the boundary so the line reaches the
        # collapse/dissolution moment instead of stopping one step early.
        cut = min(cut + 1, len(t))
        t = t[:cut]
        v2 = v2[:cut] if v2 is not None else None
        M_shell = M_shell[:cut] if M_shell is not None else None
        R = R[:cut] if R is not None else None

    return t, v2, M_shell, R


def load_simulation_at_time(data_path: Path, config: AnalysisConfig) -> Optional[SimulationResult]:
    """
    Load simulation and extract observables at specified time.
    """
    try:
        output = load_output(data_path)

        if len(output) == 0:
            return None

        # Folder-name tags drive the filename/grid layout (legitimate use
        # of the sweep convention) — kept as strings.  Scientific numerics
        # below come from metadata.json via snapshot rehydration: the
        # snapshot-side mCloud is the *post-SFE residual* (compatible with
        # ``compute_stellar_mass``'s M_star = sfe * mCloud / (1 - sfe)),
        # sfe is the exact .param value (not the 3-digit folder rounding),
        # and nCore is converted from code units back to cm^-3 to match
        # the historical axis convention.
        folder_name = data_path.parent.name
        params = parse_simulation_params(folder_name)

        if params is None:
            return None

        mCloud_str = params['mCloud']
        sfe_str = params['sfe']
        nCore_str = params['ndens']

        snap_consts = output[0]
        mCloud_float = float(snap_consts.get('mCloud'))
        sfe_float = float(snap_consts.get('sfe'))
        nCore_float = float(snap_consts.get('nCore')) * INV_CONV.ndens_au2cgs

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

        # Trim trajectories after collapse or dissolution onset so that
        # frozen/constant tails are not plotted (misleading flat lines).
        t_full, v2_full, M_shell_full, R_full = _trim_after_end(
            output, t_full, v2_full, M_shell_full, R_full)

        v_full_kms = v2_full * PC_MYR_TO_KM_S if v2_full is not None else None

        # Phase and cloud radius for markers
        phase_raw = output.get('current_phase', as_array=False)
        phase_full = None
        if phase_raw is not None:
            phase_arr = np.asarray([str(p) for p in phase_raw])
            if t_full is not None and len(phase_arr) >= len(t_full):
                phase_full = phase_arr[:len(t_full)]
        rcloud = float(output[0].get('rCloud', np.nan))

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
            phase_full=phase_full,
            rcloud=rcloud,
        )

    except Exception as e:
        print(f"  Error loading {data_path}: {e}")
        return None


def load_sweep_results(folder_path: Path, config: AnalysisConfig) -> List[SimulationResult]:
    """
    Load all simulations from a sweep folder.
    """
    folder_path = Path(folder_path)
    sim_files = find_all_simulations(folder_path)
    sim_files = filter_sim_files_by_phii(sim_files, config.phii_mode)

    if not sim_files:
        label = "non-noPHII" if config.phii_mode == "yes" else "noPHII"
        print(f"No {label} simulation files found in {folder_path}")
        return []

    results = []

    for sim_path in sim_files:
        folder_name = sim_path.parent.name
        params = parse_simulation_params(folder_name)

        if params is None:
            continue

        ndens = params['ndens']

        # Apply nCore filter if specified
        if config.nCore_filter and not nCore_matches(ndens, config.nCore_filter):
            continue

        result = load_simulation_at_time(sim_path, config)

        if result is not None:
            results.append(result)

    # Sort by chi2
    results.sort(key=lambda x: x.chi2_total)

    return results
