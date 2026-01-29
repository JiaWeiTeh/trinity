#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory evolution plots for TRINITY parameter sweeps.

This script creates shell mass and radius evolution trajectory plots
from TRINITY simulation parameter sweeps.

Produces:
- Shell mass M(t) trajectory plot
- Shell radius R(t) trajectory plot

@author: Jia Wei Teh
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Tuple
import matplotlib.colors as mcolors

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import (
    load_output, find_all_simulations,
    parse_simulation_params
)

print("...creating trajectory evolution plots")

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
    """Observational constraints for comparison."""
    # Expansion velocity
    v_obs: float = 13.0          # km/s
    v_err: float = 2.0           # km/s

    # Shell mass options
    M_shell_HI: float = 100.0        # M_sun
    M_shell_HI_err: float = 30.0     # M_sun
    M_shell_CII: float = 1000.0      # M_sun
    M_shell_CII_err: float = 300.0   # M_sun
    M_shell_combined: float = 2000.0 # M_sun
    M_shell_combined_err: float = 500.0  # M_sun

    # Dynamical age
    t_obs: float = 0.2           # Myr
    t_err: float = 0.05          # Myr

    # Shell radius
    R_obs: float = 4.0           # pc
    R_err: float = 0.5           # pc

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


# =============================================================================
# Core Functions
# =============================================================================

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


def load_simulation_at_time(data_path: Path, config: AnalysisConfig) -> Optional[SimulationResult]:
    """
    Load simulation and extract observables at specified time.
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
        sfe_float = int(sfe_str) / 100.0  # sfe is stored as percentage
        nCore_float = float(nCore_str)

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

    if not sim_files:
        print(f"No simulation files found in {folder_path}")
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


# =============================================================================
# Trajectory Plot Function
# =============================================================================

def plot_trajectory_evolution(results: List[SimulationResult], config: AnalysisConfig,
                               output_dir: Path, nCore_value: str, top_n: int = 5):
    """
    Create shell mass M(t) and radius R(t) trajectory plots.

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

    # 2 subplots: mass, radius
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), dpi=150)
    ax_m, ax_r = axes

    obs = config.obs

    # Color map for different simulations
    if config.show_all:
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
        M = r.M_shell_full
        R = r.R_full

        if config.show_all:
            color = cmap(norm(r.chi2_total))
            alpha = 0.7
            lw = 1.0
            label = None
        else:
            color = colors[i]
            alpha = 0.9
            lw = 1.5
            label = f"{r.mCloud}_sfe{r.sfe} (M$_\\star$={r.Mstar:.0f}, $\\chi^2$={r.chi2_total:.1f})"

        # Mass trajectory
        if M is not None:
            ax_m.plot(t, M, color=color, lw=lw, label=label, alpha=alpha)

        # Radius trajectory
        if R is not None:
            ax_r.plot(t, R, color=color, lw=lw, label=label, alpha=alpha)

    # --- Mass panel (log scale) ---
    tracer_bands = [
        (obs.M_shell_HI, obs.M_shell_HI_err, 'blue', r'HI ($\sim 10^2 M_\odot$)', 0.15),
        (obs.M_shell_CII, obs.M_shell_CII_err, 'darkorange', r'[CII] ($\sim 10^3 M_\odot$)', 0.15),
    ]

    for M_val, M_err, color, label, alpha in tracer_bands:
        ax_m.axhspan(M_val - M_err, M_val + M_err, alpha=alpha, color=color, zorder=1)
        ax_m.errorbar(obs.t_obs, M_val, xerr=obs.t_err, yerr=M_err,
                      fmt='s', color=color, markersize=10, capsize=4, capthick=1.5,
                      label=f'{label}', zorder=10,
                      markeredgecolor='k', markeredgewidth=0.5)

    ax_m.axvspan(obs.t_obs - obs.t_err, obs.t_obs + obs.t_err,
                 alpha=0.1, color='gray', zorder=0)

    ax_m.set_xlabel('Time [Myr]', fontsize=14)
    ax_m.set_ylabel(r'Shell Mass [$M_\odot$]', fontsize=14, rotation=90)
    ax_m.legend(loc='upper left', fontsize=7)
    ax_m.set_xlim(0, max(0.5, obs.t_obs * 2.5))
    ax_m.set_yscale('log')
    ax_m.set_ylim(10, 1e4)
    ax_m.grid(True, alpha=0.3, which='both')

    # --- Radius panel ---
    ax_r.errorbar(obs.t_obs, obs.R_obs, xerr=obs.t_err, yerr=obs.R_err,
                  fmt='s', color='green', markersize=12, capsize=5, capthick=2,
                  label=f'Observed: {obs.R_obs}±{obs.R_err} pc', zorder=10, markeredgecolor='k')
    ax_r.axvspan(obs.t_obs - obs.t_err, obs.t_obs + obs.t_err,
                 alpha=0.2, color='blue', zorder=1)
    ax_r.axhspan(obs.R_obs - obs.R_err, obs.R_obs + obs.R_err,
                 alpha=0.2, color='green', zorder=1)

    ax_r.set_xlabel('Time [Myr]', fontsize=14)
    ax_r.set_ylabel('Shell Radius [pc]', fontsize=14, rotation=90)
    ax_r.legend(loc='upper left', fontsize=7)
    ax_r.set_xlim(0, max(0.5, obs.t_obs * 2.5))
    ax_r.set_ylim(0, None)
    ax_r.grid(True, alpha=0.3)

    plt.tight_layout()

    suffix = config.get_filename_suffix()
    out_pdf = output_dir / f'trajectory_n{nCore_value}{suffix}.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


# =============================================================================
# Main Function
# =============================================================================

def main(folder_path: str, output_dir: str = None, config: AnalysisConfig = None):
    """
    Main entry point for trajectory evolution analysis.

    Parameters
    ----------
    folder_path : str
        Path to sweep output folder
    output_dir : str, optional
        Output directory for plots
    config : AnalysisConfig, optional
        Analysis configuration
    """
    if config is None:
        config = AnalysisConfig()

    # Convert to Path object
    folder_path = Path(folder_path)

    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    # Setup output directory
    if output_dir is None:
        output_dir = folder_path / "analysis"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading simulations from: {folder_path}")
    print(f"Output directory: {output_dir}")

    # Load all simulation results
    results = load_sweep_results(folder_path, config)

    if not results:
        print("No valid simulation results found.")
        sys.exit(1)

    print(f"\nLoaded {len(results)} simulations")

    # Get unique nCore values
    nCore_values = sorted(set(r.nCore for r in results), key=lambda x: float(x))
    print(f"nCore values: {nCore_values}")

    # Create trajectory plots for each nCore
    print("\nCreating trajectory evolution plots...")
    for nCore in nCore_values:
        plot_trajectory_evolution(results, config, output_dir, nCore)

    print("\nDone!")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="""TRINITY trajectory evolution plots

Creates shell mass and radius evolution trajectory plots from TRINITY
parameter sweep simulations.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic trajectory plots
  python paper_ODIN.py --folder sweep_orion/

  # Show all simulation trajectories
  python paper_ODIN.py --folder sweep_orion/ --showall

  # Filter by nCore value
  python paper_ODIN.py --folder sweep_orion/ --nCore 1e4

  # Compare all mass tracers
  python paper_ODIN.py --folder sweep_orion/ --mass-tracer all

Default Observational Constraints:
  Age:             0.2 +/- 0.05 Myr
  R_shell:         4 +/- 0.5 pc
  M_star:          34 +/- 5 M_sun

Shell Mass (shown in plots):
  M_shell (HI):    100 +/- 30 M_sun
  M_shell ([CII]): 1000 +/- 300 M_sun
  M_shell (comb):  2000 +/- 500 M_sun
        """
    )

    # Required
    parser.add_argument('--folder', '-F', required=True,
                        help='Path to sweep output folder')

    # Output
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output directory (default: {folder}/analysis/)')

    # Filter
    parser.add_argument('--nCore', '-n', default=None,
                        help='Filter by nCore value (e.g., "1e4")')

    # Constraint configuration
    parser.add_argument('--include-mshell', action='store_true',
                        help='Include shell mass in chi^2 calculation')
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
                        help='Combined shell mass [M_sun] (default: 2000.0)')
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

    # Mass tracer
    parser.add_argument('--mass-tracer', choices=['HI', 'CII', 'combined', 'all'],
                        default='combined',
                        help='Mass tracer for plots (default: combined)')
    parser.add_argument('--M-HI', type=float, default=100.0,
                        help='HI-derived shell mass [M_sun] (default: 100.0)')
    parser.add_argument('--M-HI-err', type=float, default=30.0,
                        help='HI shell mass uncertainty [M_sun] (default: 30.0)')
    parser.add_argument('--M-CII', type=float, default=1000.0,
                        help='[CII]-derived shell mass [M_sun] (default: 1000.0)')
    parser.add_argument('--M-CII-err', type=float, default=300.0,
                        help='[CII] shell mass uncertainty [M_sun] (default: 300.0)')

    # Trajectory options
    parser.add_argument('--showall', action='store_true',
                        help='Show all simulation trajectories')

    args = parser.parse_args()

    # Build observational constraints
    obs = ObservationalConstraints(
        v_obs=args.v_obs, v_err=args.v_err,
        M_shell_HI=args.M_HI, M_shell_HI_err=args.M_HI_err,
        M_shell_CII=args.M_CII, M_shell_CII_err=args.M_CII_err,
        M_shell_combined=args.M_combined, M_shell_combined_err=args.M_combined_err,
        t_obs=args.t_obs, t_err=args.t_err,
        R_obs=args.R_obs, R_err=args.R_err,
        Mstar_obs=args.Mstar, Mstar_err=args.Mstar_err,
    )

    config = AnalysisConfig(
        constrain_v=not args.no_v,
        constrain_M_shell=args.include_mshell,
        constrain_t=not args.no_t,
        constrain_R=not args.no_R,
        constrain_Mstar=not args.no_Mstar,
        nCore_filter=args.nCore,
        mass_tracer=args.mass_tracer,
        show_all=args.showall,
        obs=obs,
    )

    main(args.folder, args.output_dir, config)
