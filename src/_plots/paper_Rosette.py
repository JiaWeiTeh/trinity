#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory evolution plots for the Rosette Nebula (NGC 2244).

Analogous to paper_ODIN but with observational constraints for the
Rosette Nebula / Rosette Molecular Cloud (RMC).  Reuses data-loading
and chi² machinery from paper_ODIN; only default observational values
and plot styling differ.

Observational references
------------------------
Cluster mass       : 1000 ± 70 M☉          (Mužić et al. 2022)
Stellar population : ~2000 stars            (Wang et al. 2008, Chandra XLF)
Cluster age        : ~2 Myr                 (Mužić et al. 2022)
Distance           : 1489 ± 37 pc           (Mužić et al. 2022, Gaia EDR3)
Cavity inner radius: ~7 pc                  (Planck XXXIV, Alves et al. 2016)
HII outer radius   : ~19 pc                 (Planck XXXIV, radio emission)
Dust shell extent  : 18–22 pc               (Planck XXXIV, 353 GHz)
Shell exp. velocity: ~20–30 km/s            (Dent et al. 2009, CO J=3–2)
RMC cloud mass     : ~10⁵ M☉               (Williams, Blitz & Stark 1995)
Mean clump density : ~10³ cm⁻³             (Williams, Blitz & Stark 1995)

@author: Jia Wei Teh
"""

import sys
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src._plots.paper_ODIN import (          # noqa: E402
    ObservationalConstraints,
    AnalysisConfig,
    SimulationResult,
    smooth_trajectory,
    load_sweep_results,
    FONTSIZE,
)
from src._output.trinity_reader import info_simulations  # noqa: E402

print("...creating Rosette trajectory evolution plots")


# =============================================================================
# Rosette default observational values
# =============================================================================

def rosette_constraints() -> ObservationalConstraints:
    """Return observational constraints for the Rosette Nebula."""
    return ObservationalConstraints(
        # Expansion velocity: 20–30 km/s  (Dent et al. 2009)
        v_obs=25.0,
        v_err=5.0,
        # Shell mass: not directly measured for Rosette — disable in chi²
        M_shell_HI=0.0,
        M_shell_HI_err=0.0,
        M_shell_CII=0.0,
        M_shell_CII_err=0.0,
        M_shell_combined=0.0,
        M_shell_combined_err=0.0,
        # Cluster age: ~2 Myr  (Mužić et al. 2022)
        t_obs=2.0,
        t_err=0.5,
        # HII outer radius: ~19 pc  (Planck XXXIV, radio)
        R_obs=19.0,
        R_err=2.0,
        # Cavity inner radius: ~7 pc  (Planck XXXIV, Alves et al. 2016)
        R_obs_Pabst=7.0,
        R_err_Pabst=1.0,
        # Cluster/stellar mass: 1000 ± 70 M☉  (Mužić et al. 2022)
        Mstar_obs=1000.0,
        Mstar_err=70.0,
    )


def rosette_config(**overrides) -> AnalysisConfig:
    """Return default AnalysisConfig for the Rosette Nebula.

    Shell mass is not constrained by default (no direct observation).
    """
    defaults = dict(
        constrain_v=True,
        constrain_M_shell=False,
        constrain_t=True,
        constrain_R=True,
        constrain_Mstar=True,
        obs=rosette_constraints(),
    )
    defaults.update(overrides)
    return AnalysisConfig(**defaults)


# =============================================================================
# Plot functions  (Rosette-specific axis limits & markers)
# =============================================================================

# Rosette is much larger and older than Orion ⇒ wider axes
_T_MAX = 3.0     # Myr
_R_MAX = 30.0    # pc
_V_MAX = 60.0    # km/s

# Dust shell extent (Planck XXXIV, 353 GHz) — not part of chi²,
# shown as a visual band on the radius panel.
_DUST_SHELL_MIN = 18.0  # pc
_DUST_SHELL_MAX = 22.0  # pc


def plot_trajectory_evolution(results: List[SimulationResult],
                              config: AnalysisConfig,
                              output_dir: Path,
                              nCore_value: str,
                              top_n: int = 5):
    """Create v(t) and R(t) trajectory plots for a single nCore value."""
    data = [r for r in results if r.nCore == nCore_value]
    if not data:
        return

    data_sorted = sorted(data, key=lambda x: x.chi2_total)
    data_to_plot = data_sorted if config.show_all else data_sorted[:top_n]

    fig, (ax_v, ax_r) = plt.subplots(2, 1, figsize=(6.5, 18), dpi=150, sharex=True)
    obs = config.obs

    # Colour scheme
    if config.show_all:
        chi2_vals = [r.chi2_total for r in data_to_plot]
        norm = mcolors.LogNorm(vmin=max(0.1, min(chi2_vals)),
                               vmax=max(1, max(chi2_vals)))
        cmap = plt.cm.viridis_r
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(data_to_plot)))

    for i, r in enumerate(data_to_plot):
        if r.t_full is None:
            continue

        v_smooth = smooth_trajectory(r.t_full, r.v_full_kms)
        R_smooth = smooth_trajectory(r.t_full, r.R_full)

        if config.show_all:
            color, alpha, lw, label = cmap(norm(r.chi2_total)), 0.7, 1.0, None
        else:
            color, alpha, lw = colors[i], 0.9, 1.5
            label = (f"{r.mCloud}_sfe{r.sfe} "
                     f"(M$_\\star$={r.Mstar:.0f}, $\\chi^2$={r.chi2_total:.1f})")

        if v_smooth is not None:
            ax_v.plot(r.t_full, v_smooth, color=color, lw=lw, label=label, alpha=alpha)
        if R_smooth is not None:
            ax_r.plot(r.t_full, R_smooth, color=color, lw=lw, label=label, alpha=alpha)

    # --- Velocity panel ---
    ax_v.errorbar(obs.t_obs, obs.v_obs, xerr=obs.t_err, yerr=obs.v_err,
                  fmt='s', color='red', markersize=14, capsize=5, capthick=2,
                  label=f'$v_{{\\rm exp}}$: {obs.v_obs}\u00b1{obs.v_err} km/s', zorder=10,
                  markeredgecolor='k')
    ax_v.axhspan(obs.v_obs - obs.v_err, obs.v_obs + obs.v_err,
                 alpha=0.15, color='red', zorder=1)
    ax_v.axvspan(obs.t_obs - obs.t_err, obs.t_obs + obs.t_err,
                 alpha=0.1, color='gray', zorder=0)
    ax_v.set_ylabel(r'Expansion Velocity [km s$^{-1}$]', fontsize=FONTSIZE, rotation=90)
    ax_v.tick_params(axis='both', labelsize=FONTSIZE)
    ax_v.tick_params(axis='y', labelrotation=90)
    ax_v.legend(loc='upper right', fontsize=FONTSIZE).set_zorder(100)
    ax_v.set_ylim(0, _V_MAX)
    ax_v.grid(True, alpha=0.3)
    ax_v.tick_params(axis='x', pad=10)
    ax_v.tick_params(axis='y', pad=10)

    # --- Radius panel ---
    # HII outer radius (blue)
    ax_r.errorbar(obs.t_obs, obs.R_obs, xerr=obs.t_err, yerr=obs.R_err,
                  fmt='s', color='blue', markersize=14, capsize=5, capthick=2,
                  label=f'HII outer: {obs.R_obs}\u00b1{obs.R_err} pc', zorder=10,
                  markeredgecolor='k')
    ax_r.axhspan(obs.R_obs - obs.R_err, obs.R_obs + obs.R_err,
                 alpha=0.15, color='blue', zorder=1)

    # Cavity inner radius (green)
    ax_r.errorbar(obs.t_obs, obs.R_obs_Pabst, xerr=obs.t_err, yerr=obs.R_err_Pabst,
                  fmt='s', color='green', markersize=14, capsize=5, capthick=2,
                  label=f'Cavity: {obs.R_obs_Pabst}\u00b1{obs.R_err_Pabst} pc', zorder=10,
                  markeredgecolor='k')
    ax_r.axhspan(obs.R_obs_Pabst - obs.R_err_Pabst,
                 obs.R_obs_Pabst + obs.R_err_Pabst,
                 alpha=0.15, color='green', zorder=1)

    # Dust shell extent band (orange)
    ax_r.axhspan(_DUST_SHELL_MIN, _DUST_SHELL_MAX, alpha=0.12, color='orange',
                 zorder=1,
                 label=f'Dust shell ({_DUST_SHELL_MIN:.0f}\u2013{_DUST_SHELL_MAX:.0f} pc)')

    ax_r.axvspan(obs.t_obs - obs.t_err, obs.t_obs + obs.t_err,
                 alpha=0.1, color='gray', zorder=0)

    ax_r.set_xlabel('Time [Myr]', fontsize=FONTSIZE)
    ax_r.set_ylabel('Shell Radius [pc]', fontsize=FONTSIZE, rotation=90)
    ax_r.tick_params(axis='both', labelsize=FONTSIZE)
    ax_r.tick_params(axis='y', labelrotation=90)
    ax_r.legend(loc='lower right', fontsize=FONTSIZE).set_zorder(100)
    ax_r.set_xlim(0, _T_MAX)
    ax_r.set_ylim(0, _R_MAX)
    ax_r.grid(True, alpha=0.3)
    ax_r.tick_params(axis='x', pad=10)
    ax_r.tick_params(axis='y', pad=10)

    plt.tight_layout()
    suffix = config.get_filename_suffix()
    out_pdf = output_dir / f'rosette_trajectory_n{nCore_value}{suffix}.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


def plot_trajectory_evolution_combined(results: List[SimulationResult],
                                       config: AnalysisConfig,
                                       output_dir: Path,
                                       nCore_values: List[str],
                                       top_n: int = 5):
    """Combined trajectory plot with multiple nCore values as shaded envelopes."""
    from scipy.interpolate import interp1d

    nCore_colors = ['orange', 'r', 'darkgray', 'C3', 'C4', 'C5']
    nCore_alphas = [0.7, 0.45, 0.85, 0.7, 0.7, 0.7]

    fig, (ax_v, ax_r) = plt.subplots(2, 1, figsize=(6.5, 6.5), dpi=150, sharex=True)
    obs = config.obs
    t_common = np.linspace(0, _T_MAX, 500)

    for idx, nCore_value in enumerate(nCore_values):
        data = [r for r in results if r.nCore == nCore_value]
        if not data:
            continue

        data_sorted = sorted(data, key=lambda x: x.chi2_total)
        data_to_plot = data_sorted if config.show_all else data_sorted[:top_n]
        color = nCore_colors[idx % len(nCore_colors)]
        band_alpha = nCore_alphas[idx % len(nCore_alphas)]

        v_interp, R_interp = [], []
        for r in data_to_plot:
            if r.t_full is None or len(r.t_full) < 2:
                continue
            v_s = smooth_trajectory(r.t_full, r.v_full_kms)
            R_s = smooth_trajectory(r.t_full, r.R_full)
            try:
                if v_s is not None:
                    v_interp.append(interp1d(r.t_full, v_s, bounds_error=False,
                                             fill_value=np.nan)(t_common))
                if R_s is not None:
                    R_interp.append(interp1d(r.t_full, R_s, bounds_error=False,
                                             fill_value=np.nan)(t_common))
            except Exception:
                pass

        ncore_lbl = r'$n_{\rm core} = $' + f'{float(nCore_value):g}' + r' $\rm cm^{-3}$'
        if v_interp:
            v_arr = np.array(v_interp)
            ax_v.fill_between(t_common, np.nanmin(v_arr, 0), np.nanmax(v_arr, 0),
                              alpha=band_alpha, color=color, label=ncore_lbl)
        if R_interp:
            R_arr = np.array(R_interp)
            ax_r.fill_between(t_common, np.nanmin(R_arr, 0), np.nanmax(R_arr, 0),
                              alpha=band_alpha, color=color)

    # --- Velocity panel ---
    ax_v.errorbar(obs.t_obs, obs.v_obs, xerr=obs.t_err, yerr=obs.v_err,
                  fmt='s', color='red', markersize=10, capsize=4, capthick=1.5,
                  label=f'$v_{{\\rm exp}}$: {obs.v_obs}\u00b1{obs.v_err} km/s', zorder=10,
                  markeredgecolor='k', markeredgewidth=0.5)
    ax_v.axhspan(obs.v_obs - obs.v_err, obs.v_obs + obs.v_err,
                 alpha=0.15, color='red', zorder=1)
    ax_v.axvspan(obs.t_obs - obs.t_err, obs.t_obs + obs.t_err,
                 alpha=0.1, color='gray', zorder=0)

    # Separate nCore handles from observational handles
    all_handles, all_labels = ax_v.get_legend_handles_labels()
    ncore_handles, ncore_labels = [], []
    obs_handles, obs_labels = [], []
    for h, l in zip(all_handles, all_labels):
        if r'n_{\rm core}' in l:
            ncore_handles.append(h)
            ncore_labels.append(l)
        else:
            obs_handles.append(h)
            obs_labels.append(l)

    # Observational legend inside velocity panel
    ax_v.set_ylabel(r'Expansion Velocity [km s$^{-1}$]', fontsize=FONTSIZE, rotation=90)
    ax_v.tick_params(axis='both', labelsize=FONTSIZE)
    ax_v.tick_params(axis='y', labelrotation=90)
    if obs_handles:
        legend_v = ax_v.legend(obs_handles, obs_labels, loc='upper left',
                               fontsize=FONTSIZE)
        legend_v.set_zorder(100)
    ax_v.set_ylim(0, _V_MAX)
    ax_v.grid(True, alpha=0.3)

    # nCore legend placed above the top subplot as a figure-level title legend
    if ncore_handles:
        fig.legend(ncore_handles, ncore_labels,
                   loc='upper center', ncol=len(ncore_handles),
                   fontsize=FONTSIZE, frameon=False,
                   bbox_to_anchor=(0.5, 1.1))

    # --- Radius panel ---
    ax_r.errorbar(obs.t_obs, obs.R_obs, xerr=obs.t_err, yerr=obs.R_err,
                  fmt='s', color='blue', markersize=12, capsize=5, capthick=2,
                  label=f'HII outer: {obs.R_obs}\u00b1{obs.R_err} pc', zorder=10,
                  markeredgecolor='k')
    ax_r.axhspan(obs.R_obs - obs.R_err, obs.R_obs + obs.R_err,
                 alpha=0.15, color='blue', zorder=1)

    ax_r.errorbar(obs.t_obs, obs.R_obs_Pabst, xerr=obs.t_err, yerr=obs.R_err_Pabst,
                  fmt='s', color='green', markersize=12, capsize=5, capthick=2,
                  label=f'Cavity: {obs.R_obs_Pabst}\u00b1{obs.R_err_Pabst} pc', zorder=10,
                  markeredgecolor='k')
    ax_r.axhspan(obs.R_obs_Pabst - obs.R_err_Pabst,
                 obs.R_obs_Pabst + obs.R_err_Pabst,
                 alpha=0.15, color='green', zorder=1)

    # Dust shell extent band (orange)
    ax_r.axhspan(_DUST_SHELL_MIN, _DUST_SHELL_MAX, alpha=0.12, color='orange',
                 zorder=1,
                 label=f'Dust shell ({_DUST_SHELL_MIN:.0f}\u2013{_DUST_SHELL_MAX:.0f} pc)')

    ax_r.axvspan(obs.t_obs - obs.t_err, obs.t_obs + obs.t_err,
                 alpha=0.1, color='gray', zorder=0)

    ax_r.set_xlabel('Time [Myr]', fontsize=FONTSIZE)
    ax_r.set_ylabel('Shell Radius [pc]', fontsize=FONTSIZE, rotation=90)
    ax_r.tick_params(axis='both', labelsize=FONTSIZE)
    ax_r.tick_params(axis='y', labelrotation=90)
    legend_r = ax_r.legend(loc='upper left', fontsize=FONTSIZE)
    legend_r.set_zorder(100)
    ax_r.set_xlim(0, _T_MAX)
    ax_r.set_ylim(0, _R_MAX)
    ax_r.grid(True, alpha=0.3)

    plt.tight_layout(h_pad=1.0)
    suffix = config.get_filename_suffix()
    nCore_str = "_".join(nCore_values)
    out_pdf = output_dir / f'rosette_trajectory_n{nCore_str}{suffix}.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main(folder_path: str, output_dir: str = None, config: AnalysisConfig = None):
    if config is None:
        config = rosette_config()

    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    if output_dir is None:
        output_dir = folder_path / "analysis"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading simulations from: {folder_path}")
    print(f"Output directory: {output_dir}")

    results = load_sweep_results(folder_path, config)
    if not results:
        print("No valid simulation results found.")
        sys.exit(1)

    print(f"\nLoaded {len(results)} simulations")

    nCore_values = sorted(set(r.nCore for r in results), key=lambda x: float(x))
    print(f"nCore values: {nCore_values}")

    print("\nCreating Rosette trajectory evolution plots...")
    if config.combine_nCore and len(nCore_values) > 1:
        plot_trajectory_evolution_combined(results, config, output_dir, nCore_values)
    else:
        for nCore in nCore_values:
            plot_trajectory_evolution(results, config, output_dir, nCore)

    print("\nDone!")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TRINITY trajectory evolution plots — Rosette Nebula",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_Rosette.py --folder sweep_rosette/
  python paper_Rosette.py --folder sweep_rosette/ --showall
  python paper_Rosette.py --folder sweep_rosette/ --nCore 1e3
  python paper_Rosette.py --folder sweep_rosette/ --combine-nCore

Default Observational Constraints (Rosette Nebula):
  Age (cluster)      : 2.0 +/- 0.5 Myr     (Muzic et al. 2022)
  Expansion velocity : 25 +/- 5 km/s        (Dent et al. 2009)
  HII outer radius   : 19 +/- 2 pc          (Planck XXXIV)
  Cavity inner radius: 7 +/- 1 pc           (Planck XXXIV / Alves+16)
  Dust shell extent  : 18–22 pc             (Planck XXXIV, 353 GHz)
  Cluster mass       : 1000 +/- 70 M_sun    (Muzic et al. 2022)
        """
    )

    # Required
    parser.add_argument('--folder', '-F', required=True,
                        help='Path to sweep output folder')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output directory (default: {folder}/analysis/)')

    # Filters
    parser.add_argument('--nCore', '-n', default=None,
                        help='Filter by nCore value (e.g., "1e3")')
    parser.add_argument('--info', action='store_true',
                        help='Print available parameter values and exit')

    # Constraint toggles
    parser.add_argument('--include-mshell', action='store_true',
                        help='Include shell mass in chi² (off by default)')
    parser.add_argument('--no-Mstar', action='store_true',
                        help='Exclude stellar/cluster mass constraint')
    parser.add_argument('--no-v', action='store_true',
                        help='Exclude velocity constraint')
    parser.add_argument('--no-t', action='store_true',
                        help='Exclude age constraint')
    parser.add_argument('--no-R', action='store_true',
                        help='Exclude shell radius constraint')

    # Custom observational overrides (Rosette defaults)
    parser.add_argument('--v-obs', type=float, default=25.0,
                        help='Expansion velocity [km/s] (default: 25)')
    parser.add_argument('--v-err', type=float, default=5.0,
                        help='Velocity uncertainty [km/s] (default: 5)')
    parser.add_argument('--t-obs', type=float, default=2.0,
                        help='Cluster age [Myr] (default: 2.0)')
    parser.add_argument('--t-err', type=float, default=0.5,
                        help='Age uncertainty [Myr] (default: 0.5)')
    parser.add_argument('--R-obs', type=float, default=19.0,
                        help='HII outer radius [pc] (default: 19)')
    parser.add_argument('--R-err', type=float, default=2.0,
                        help='Radius uncertainty [pc] (default: 2)')
    parser.add_argument('--R-cavity', type=float, default=7.0,
                        help='Cavity inner radius [pc] (default: 7)')
    parser.add_argument('--R-cavity-err', type=float, default=1.0,
                        help='Cavity radius uncertainty [pc] (default: 1)')
    parser.add_argument('--Mstar', type=float, default=1000.0,
                        help='Cluster mass [M_sun] (default: 1000)')
    parser.add_argument('--Mstar-err', type=float, default=70.0,
                        help='Cluster mass uncertainty [M_sun] (default: 70)')

    # Trajectory options
    parser.add_argument('--showall', action='store_true',
                        help='Show all simulation trajectories')
    parser.add_argument('--combine-nCore', action='store_true',
                        help='Plot all nCore values on the same plot')
    parser.add_argument('--mass-tracer', choices=['HI', 'CII', 'combined', 'all'],
                        default='combined',
                        help='Mass tracer for chi² (default: combined)')

    args = parser.parse_args()

    # --info mode
    if args.info:
        info = info_simulations(args.folder)
        print("=" * 50)
        print(f"Simulation parameters in: {args.folder}")
        print("=" * 50)
        print(f"  Total simulations: {info['count']}")
        print(f"  mCloud values: {info['mCloud']}")
        print(f"  SFE values: {info['sfe']}")
        print(f"  nCore values: {info['ndens']}")
        sys.exit(0)

    # Build config
    obs = ObservationalConstraints(
        v_obs=args.v_obs, v_err=args.v_err,
        M_shell_HI=0.0, M_shell_HI_err=0.0,
        M_shell_CII=0.0, M_shell_CII_err=0.0,
        M_shell_combined=0.0, M_shell_combined_err=0.0,
        t_obs=args.t_obs, t_err=args.t_err,
        R_obs=args.R_obs, R_err=args.R_err,
        R_obs_Pabst=args.R_cavity, R_err_Pabst=args.R_cavity_err,
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
        combine_nCore=args.combine_nCore,
        obs=obs,
    )

    main(args.folder, args.output_dir, config)
