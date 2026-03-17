#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zeta regime map: ζ = R_eq / R_St across (n_cloud, M_cl) parameter space.

ζ > 1 → bubble-dominated (wind drives expansion)
ζ < 1 → HII-dominated (photoionisation drives expansion)

Two modes:
  Mode B (analytic): smooth background from power-law Qi(M_cl), pdot_w(M_cl)
  Mode A (from sims): scatter overlay with measured ζ at T_REF

Reference: Lancaster et al. (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from src._plots.plot_base import FIG_DIR
from src._plots.force_colors import C  # noqa: E402
from src._functions.unit_conversions import CGS, CONV, INV_CONV

print("...plotting zeta regime map")

# ======================================================================
# Configuration
# ======================================================================
T_REF = 1.0          # Myr — reference time for ζ evaluation
N_GRID = 200         # resolution for analytic grid
LOG_N_RANGE = (1, 5)       # log10(n [cm⁻³])
LOG_MCL_RANGE = (2, 7)     # log10(M_cl [M_☉])
SAVE_PDF = True

# Physical constants (CGS)
ALPHA_B = 2.56e-13       # Case-B recombination coefficient [cm³ s⁻¹] at T=10⁴ K
T_ION = 1e4              # Ionised gas temperature [K]
MU_ION = 0.678           # Mean molecular weight (ionised)
MU_NEUTRAL = 1.27        # Mean molecular weight (neutral, for ρ = n μ m_H)

# Derived: isothermal sound speed squared in ionised gas
# c_i² = k_B T_ion / (μ_ion m_H)   [cm² s⁻²]
C_I_SQ = CGS.k_B * T_ION / (MU_ION * CGS.m_H)  # cm² s⁻²


# ======================================================================
# Analytic power-law scalings (Mode B)
# ======================================================================
# For a simple stellar population at ~1 Myr (main-sequence dominated):
#   Qi ≈ 10^{46.5} × (M_cl / M_☉) s⁻¹  (Leitherer+ 1999, SB99 tables)
#   pdot_w ≈ 10^{23} × (M_cl / M_☉) g cm s⁻² (typical O-star winds)
#
# These are order-of-magnitude calibrations; the exact values depend on
# metallicity, IMF, and stellar tracks.  The _shape_ of the ζ map (the
# ζ=1 contour slope) is robust because ζ ∝ M_cl^{1/6} n^{1/3}.

# Default calibration at T_REF ≈ 1 Myr, solar Z, Geneva rot tracks
QI_PER_MSUN = 10**46.5    # s⁻¹ per M_☉ of cluster mass
PDOT_W_PER_MSUN = 10**23  # g cm s⁻² per M_☉ of cluster mass


def compute_zeta_analytic(log_Mcl_arr, log_n_arr,
                          Qi_per_Msun=QI_PER_MSUN,
                          pdot_w_per_Msun=PDOT_W_PER_MSUN):
    """
    Compute ζ = R_eq / R_St on a 2D grid (analytic, Mode B).

    Parameters
    ----------
    log_Mcl_arr : 1-D array
        log10(M_cl / M_☉)
    log_n_arr : 1-D array
        log10(n / cm⁻³)
    Qi_per_Msun, pdot_w_per_Msun : float
        Normalisation per solar mass of cluster.

    Returns
    -------
    zeta : 2-D array, shape (len(log_Mcl_arr), len(log_n_arr))
        ζ values.  zeta[i, j] corresponds to (log_Mcl_arr[i], log_n_arr[j]).
    """
    Mcl = 10.0**log_Mcl_arr                   # M_☉
    n = 10.0**log_n_arr                        # cm⁻³

    # Broadcast: Mcl along axis 0, n along axis 1
    Mcl_2d = Mcl[:, None]
    n_2d = n[None, :]

    Qi = Qi_per_Msun * Mcl_2d                  # s⁻¹
    pdot_w = pdot_w_per_Msun * Mcl_2d          # g cm s⁻²

    # Strömgren radius
    R_St = (3.0 * Qi / (4.0 * np.pi * ALPHA_B * n_2d**2))**(1.0 / 3.0)  # cm

    # Wind equilibrium radius
    rho = n_2d * MU_NEUTRAL * CGS.m_H         # g cm⁻³
    R_eq = np.sqrt(3.0 * pdot_w / (16.0 * np.pi * rho * C_I_SQ))  # cm

    zeta = R_eq / R_St
    return zeta


# ======================================================================
# Mode A — extract ζ from simulation snapshots
# ======================================================================

def compute_zeta_from_sims(folder_path, t_ref=T_REF):
    """
    Load simulations and compute ζ at t_ref for each run.

    Returns
    -------
    results : list of dict
        Each dict: {'mCloud': float, 'mCluster': float, 'n_cloud': float,
                    'sfe': float, 'zeta': float}
    """
    from src._output.trinity_reader import (
        load_output, find_all_simulations, parse_simulation_params
    )

    folder_path = Path(folder_path)
    sim_files = find_all_simulations(folder_path)
    if not sim_files:
        print(f"  No simulations found in {folder_path}")
        return []

    results = []
    for data_path in sim_files:
        folder_name = data_path.parent.name
        sim_params = parse_simulation_params(folder_name)
        if sim_params is None:
            continue

        try:
            output = load_output(data_path)
            if len(output) == 0:
                continue

            # Get snapshot closest to t_ref
            snap = output.get_at_time(t_ref)
            if snap is None:
                continue

            # Extract quantities from snapshot
            Qi = snap.get('Qi')             # s⁻¹ (AU-converted in output)
            pdot_W = snap.get('pdot_W')     # Msun pc / Myr²
            mCloud_val = snap.get('mCloud')
            nEdge = snap.get('nEdge')       # cm⁻³ (ambient density)

            if any(v is None for v in [Qi, pdot_W, mCloud_val, nEdge]):
                continue
            if nEdge <= 0 or Qi <= 0 or pdot_W <= 0:
                continue

            # Qi is stored in AU (photons/Myr) — convert back to s⁻¹
            Qi_cgs = float(Qi) * CONV.s2Myr  # photons/Myr × (s/Myr) = photons/s? No.
            # Actually: output stores Qi with unit conversion s2Myr applied as:
            #   Qi_AU = Qi_cgs / s2Myr  (since rate = 1/s → 1/Myr)
            # So to get back to cgs: Qi_cgs = Qi_AU / s2Myr? No.
            # From read_SB99: Qi = 10^col * f_mass / s2Myr
            # i.e., Qi_stored = Qi_cgs_per_s / s2Myr = Qi_cgs × Myr2s
            # To convert back: Qi_cgs = Qi_stored × s2Myr
            # Wait: 1/s → 1/Myr means multiply by Myr2s = 1/s2Myr
            # So Qi_stored [1/Myr] = Qi_cgs [1/s] × Myr2s = Qi_cgs / s2Myr
            # Therefore Qi_cgs = Qi_stored × s2Myr
            Qi_cgs = float(Qi) * CONV.s2Myr  # back to s⁻¹

            # pdot_W is stored in AU (Msun pc / Myr²) — convert to cgs (g cm / s²)
            pdot_w_cgs = float(pdot_W) * INV_CONV.pdot_au2cgs

            n_cloud = float(nEdge)  # cm⁻³

            # Compute ζ
            R_St = (3.0 * Qi_cgs / (4.0 * np.pi * ALPHA_B * n_cloud**2))**(1.0 / 3.0)
            rho = n_cloud * MU_NEUTRAL * CGS.m_H
            R_eq = np.sqrt(3.0 * pdot_w_cgs / (16.0 * np.pi * rho * C_I_SQ))
            zeta = R_eq / R_St

            sfe_val = float(sim_params['sfe']) / 100.0
            mCluster = float(mCloud_val) * sfe_val

            results.append({
                'mCloud': float(mCloud_val),
                'mCluster': mCluster,
                'n_cloud': n_cloud,
                'sfe': sfe_val,
                'zeta': zeta,
            })

        except Exception as e:
            print(f"  Warning: could not process {folder_name}: {e}")
            continue

    print(f"  Extracted ζ from {len(results)} simulations")
    return results


# ======================================================================
# Plotting
# ======================================================================

def plot_zeta_regime(folder_path=None, output_dir=None, t_ref=T_REF):
    """
    Create the ζ regime map figure.

    Parameters
    ----------
    folder_path : str or Path, optional
        Simulation sweep folder (Mode A overlay).  If None, analytic only.
    output_dir : str or Path, optional
        Output directory for PDF.
    t_ref : float
        Reference time in Myr.
    """

    # --- Mode B: analytic background ---
    log_n = np.linspace(*LOG_N_RANGE, N_GRID)
    log_Mcl = np.linspace(*LOG_MCL_RANGE, N_GRID)
    zeta = compute_zeta_analytic(log_Mcl, log_n)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)

    # Diverging colourmap centred at ζ = 1 (log-scale)
    # ζ range roughly 0.01 to 100 → log10(ζ) in [-2, 2]
    norm = mcolors.LogNorm(vmin=0.01, vmax=100, clip=True)

    # Use a diverging colourmap: blue = HII-dominated, red = bubble-dominated
    cmap = plt.cm.RdBu_r.copy()

    im = ax.pcolormesh(
        log_n, log_Mcl, zeta,
        cmap=cmap, norm=norm, shading='auto', rasterized=True, zorder=1
    )

    # --- ζ = 1 contour (key result) ---
    n_grid, Mcl_grid = np.meshgrid(log_n, log_Mcl)
    cs1 = ax.contour(
        n_grid, Mcl_grid, zeta,
        levels=[1.0], colors='black', linewidths=2.5, zorder=3
    )
    ax.clabel(cs1, fmt={1.0: r'$\zeta = 1$'}, fontsize=10, inline=True)

    # Context contours at ζ = 0.1 and ζ = 10
    cs_ctx = ax.contour(
        n_grid, Mcl_grid, zeta,
        levels=[0.1, 10.0], colors='black', linewidths=0.8,
        linestyles='--', zorder=3
    )
    ax.clabel(cs_ctx, fmt={0.1: r'$\zeta = 0.1$', 10.0: r'$\zeta = 10$'},
              fontsize=8, inline=True)

    # --- Mode A: simulation overlay ---
    sim_results = []
    if folder_path is not None:
        sim_results = compute_zeta_from_sims(folder_path, t_ref=t_ref)

    if sim_results:
        n_vals = np.array([r['n_cloud'] for r in sim_results])
        Mcl_vals = np.array([r['mCluster'] for r in sim_results])
        z_vals = np.array([r['zeta'] for r in sim_results])
        sfe_vals = np.array([r['sfe'] for r in sim_results])

        # Marker size encodes SFE
        sizes = 30 + 200 * sfe_vals  # sfe 0.01 → ~32, sfe 0.2 → ~70

        sc = ax.scatter(
            np.log10(n_vals), np.log10(Mcl_vals),
            c=z_vals, cmap=cmap, norm=norm,
            s=sizes, edgecolors='black', linewidths=0.8, zorder=5,
            marker='o'
        )

    # --- Observational reference regions ---
    # PHANGS typical GMCs: n ~ 10²–10³, M_cl ~ 10³–10⁵
    rect_phangs = plt.Rectangle(
        (2.0, 3.0), 1.0, 2.0,
        fill=False, edgecolor='0.3', linewidth=1.2, linestyle='--',
        zorder=4, label='PHANGS typical'
    )
    ax.add_patch(rect_phangs)
    ax.text(2.5, 5.15, 'PHANGS', ha='center', va='bottom',
            fontsize=7, color='0.3', style='italic')

    # --- Region labels ---
    ax.text(0.05, 0.92, r'HII-dominated ($\zeta < 1$)',
            transform=ax.transAxes, fontsize=10, color=C.DRIVE,
            fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='none', alpha=0.8))
    ax.text(0.95, 0.08, r'Bubble-dominated ($\zeta > 1$)',
            transform=ax.transAxes, fontsize=10, color=C.PHII,
            fontweight='bold', va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='none', alpha=0.8))

    # --- Colourbar ---
    cbar = fig.colorbar(im, ax=ax, label=r'$\zeta = R_{\rm eq} / R_{\rm St}$',
                        pad=0.02)
    cbar.ax.axhline(1.0, color='black', linewidth=1.5)

    # --- Axes ---
    ax.set_xlabel(r'$\log_{10}(n_{\rm cloud}\;[\mathrm{cm}^{-3}])$')
    ax.set_ylabel(r'$\log_{10}(M_{\rm cl}\;[M_\odot])$')
    ax.set_title(rf'$\zeta$ regime at $t = {t_ref:.1f}$ Myr'
                 r'  ($\zeta = R_{\rm eq}/R_{\rm St}$, Lancaster+2025)')

    ax.set_xlim(*LOG_N_RANGE)
    ax.set_ylim(*LOG_MCL_RANGE)

    plt.tight_layout()

    # --- Save ---
    if folder_path is not None:
        folder_name = Path(folder_path).name
        fig_dir = Path(output_dir) if output_dir else FIG_DIR / folder_name
    else:
        fig_dir = Path(output_dir) if output_dir else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_PDF:
        tag = f"_{Path(folder_path).name}" if folder_path else "_analytic"
        out_pdf = fig_dir / f"zetaRegime{tag}.pdf"
        fig.savefig(out_pdf, bbox_inches='tight')
        print(f"Saved: {out_pdf}")

    plt.close(fig)


# ======================================================================
# CLI wrappers (compatible with shared dispatch)
# ======================================================================

def plot_from_path(data_input: str, output_dir: str = None):
    """Single-simulation mode: not applicable. Show analytic map instead."""
    plot_zeta_regime(folder_path=None, output_dir=output_dir)


def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None):
    """
    Grid mode: analytic background + simulation overlay.

    The ndens/mCloud/sfe filters are accepted for CLI compatibility but
    not used (all simulations contribute to the scatter overlay).
    """
    plot_zeta_regime(folder_path=folder_path, output_dir=output_dir)


if __name__ == "__main__":
    from src._plots.cli import dispatch
    dispatch(
        script_name="paper_zetaRegime.py",
        description="Plot TRINITY zeta regime map",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
    )
