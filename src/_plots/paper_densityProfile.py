#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Density Profile Comparison Diagnostics for TRINITY.

Compares four density profiles (same cloud mass, SFE, core density, but varying
density structure) from a density_profile_sweep run:
  - Power-law alpha = 0  (uniform)
  - Power-law alpha = -1
  - Power-law alpha = -2 (singular isothermal sphere)
  - Bonnor-Ebert with Omega = 14.1

Produces 8 diagnostic figures examining how cloud density structure affects
feedback-driven shell evolution.

Usage:
  python paper_densityProfile.py -F <path_to_density_profile_sweep_output>
  python paper_densityProfile.py -F outputs/density_profile_sweep --fmt png
  python paper_densityProfile.py -F outputs/density_profile_sweep --show

@author: Jia Wei Teh
"""

import sys
import os
import re
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import (
    load_output, find_all_simulations, TrinityOutput
)
from src._functions.unit_conversions import CONV, INV_CONV, CGS
from src.cloud_properties.bonnorEbertSphere import (
    solve_lane_emden, create_BE_sphere
)
from src.cloud_properties.powerLawSphere import (
    compute_rCloud_homogeneous, compute_rCloud_powerlaw
)
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Load matplotlib style
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))

# =============================================================================
# Constants
# =============================================================================

# Colourblind-safe palette (Wong 2011) with solid lines for clarity
PROFILE_STYLES = {
    'PL0':  {'color': '#0072B2', 'ls': '-',  'label': r'$\alpha=0$ (uniform)'},
    'PL-1': {'color': '#D55E00', 'ls': '-',  'label': r'$\alpha=-1$'},
    'PL-2': {'color': '#009E73', 'ls': '-',  'label': r'$\alpha=-2$ (SIS)'},
    'BE14': {'color': '#CC79A7', 'ls': '-',  'label': r'BE $\Omega=14.1$'},
}

# Ordered list for consistent iteration
PROFILE_ORDER = ['PL0', 'PL-1', 'PL-2', 'BE14']

# Boltzmann constant in CGS
K_B_CGS = CGS.k_B  # erg/K

# Velocity conversion: pc/Myr -> km/s
V_AU2KMS = INV_CONV.v_au2kms

# Force conversion: AU -> CGS (dyn)
F_AU2CGS = INV_CONV.F_au2cgs

# Pressure conversion: AU -> CGS (dyn/cm^2)
PB_AU2CGS = INV_CONV.Pb_au2cgs

# Output figure directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Identification of density profile runs
# =============================================================================

def identify_profile_tag(folder_name: str) -> str:
    """
    Identify the density profile tag from a folder name.

    Parameters
    ----------
    folder_name : str
        Simulation folder name, e.g. '1e5_sfe001_n1e4_PL0' or '1e5_sfe001_n1e4_BE14'

    Returns
    -------
    str or None
        Profile tag like 'PL0', 'PL-1', 'PL-2', 'BE14', or None if not recognized.
    """
    # Match _PL{int} or _BE{int} at end of folder name
    match = re.search(r'_(PL-?\d+|BE\d+)$', folder_name)
    if match:
        return match.group(1)
    return None


def load_sweep_simulations(sweep_dir: str) -> dict:
    """
    Load all simulations from a density profile sweep directory.

    Parameters
    ----------
    sweep_dir : str
        Path to the sweep output directory.

    Returns
    -------
    dict
        Maps profile tag (e.g. 'PL0', 'BE14') -> TrinityOutput object.
    """
    sweep_path = Path(sweep_dir)
    sim_files = find_all_simulations(sweep_path)

    if not sim_files:
        raise FileNotFoundError(f"No simulations found in {sweep_path}")

    simulations = {}
    for data_path in sim_files:
        folder_name = data_path.parent.name
        tag = identify_profile_tag(folder_name)
        if tag is None:
            logger.warning(f"Could not identify profile from folder: {folder_name}")
            continue

        if tag not in PROFILE_STYLES:
            logger.warning(f"Unknown profile tag '{tag}' from folder: {folder_name}")
            continue

        logger.info(f"Loading {tag}: {data_path}")
        output = load_output(data_path)
        simulations[tag] = output

    loaded_tags = list(simulations.keys())
    logger.info(f"Loaded {len(simulations)} simulations: {loaded_tags}")
    return simulations


# =============================================================================
# Helper functions
# =============================================================================

def safe_get(output: TrinityOutput, key: str, default_val: float = 0.0) -> np.ndarray:
    """Get a field from TrinityOutput, returning default array if missing."""
    try:
        values = output.get(key, as_array=False)
    except Exception:
        return np.full(len(output), default_val)
    if values is None:
        return np.full(len(output), default_val)
    # Convert to float array, replacing None with default
    result = np.array(
        [default_val if v is None else float(v) for v in values],
        dtype=float
    )
    return np.where(np.isfinite(result), result, default_val)


def get_style(tag: str) -> dict:
    """Get plotting style for a profile tag."""
    return PROFILE_STYLES.get(tag, {'color': 'gray', 'ls': '-', 'label': tag})


def add_legend(ax, tags: list, extra_handles: list = None, **kwargs):
    """Add a consistent legend for profile tags."""
    handles = []
    for tag in tags:
        s = get_style(tag)
        handles.append(Line2D([0], [0], color=s['color'], ls=s['ls'], lw=1.5,
                              label=s['label']))
    if extra_handles:
        handles.extend(extra_handles)
    ax.legend(handles=handles, **kwargs)


def savefig(fig, name: str, output_dir: Path, fmt: str = 'pdf'):
    """Save figure with standard naming convention."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{name}.{fmt}"
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    logger.info(f"Saved: {filepath}")
    return filepath


# =============================================================================
# Helpers: read initial-condition parameters from _summary.txt
# =============================================================================

# Default fallback values (matching density_profile_sweep.param)
_DEFAULTS = dict(
    mCloud=1e5 * (1 - 0.01),   # Msun, post-SFE
    nCore=1e4 * CONV.ndens_cgs2au,  # 1/pc³
    rCore=1.0,                  # pc
    nISM=0.1 * CONV.ndens_cgs2au,   # 1/pc³
    mu_ion=1.4 * CGS.m_H * CONV.g2Msun,  # Msun
    dens_profile='densPL',
    densPL_alpha=0.0,
    densBE_Omega=14.1,
)


def parse_summary_file(summary_path: Path) -> dict:
    """
    Parse a TRINITY ``_summary.txt`` file into a {key: value_string} dict.

    Format produced by ``read_param.py``:
        key (left-padded to 30 chars)  value_repr
    Lines starting with ``#`` are comments.
    """
    result = {}
    with open(summary_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # First token is the key, remainder is the value string
            parts = line.split(None, 1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


def _try_float(val_str: str, fallback: float = 0.0) -> float:
    """Safely convert a string to float."""
    try:
        return float(val_str)
    except (ValueError, TypeError):
        return fallback


def get_cloud_params(sim_folder: Path) -> dict:
    """
    Read cloud initial-condition parameters for a simulation.

    Looks for ``*_summary.txt`` in *sim_folder* (same directory as
    ``dictionary.jsonl``).  Falls back to hard-coded defaults if the
    file is missing.

    Returns a dict with keys:
        mCloud, nCore, rCore, nISM, mu_ion   – all in internal units
        dens_profile, densPL_alpha, densBE_Omega
    """
    # Look for _summary.txt alongside dictionary.jsonl
    summaries = sorted(sim_folder.glob('*_summary.txt'))
    if not summaries:
        logger.warning(f"No _summary.txt in {sim_folder}; using defaults")
        return dict(_DEFAULTS)

    raw = parse_summary_file(summaries[0])

    return {
        'mCloud':        _try_float(raw.get('mCloud'),        _DEFAULTS['mCloud']),
        'nCore':         _try_float(raw.get('nCore'),         _DEFAULTS['nCore']),
        'rCore':         _try_float(raw.get('rCore'),         _DEFAULTS['rCore']),
        'nISM':          _try_float(raw.get('nISM'),          _DEFAULTS['nISM']),
        'mu_ion':        _try_float(raw.get('mu_ion'),        _DEFAULTS['mu_ion']),
        'dens_profile':  raw.get('dens_profile',              _DEFAULTS['dens_profile']),
        'densPL_alpha':  _try_float(raw.get('densPL_alpha'),  _DEFAULTS['densPL_alpha']),
        'densBE_Omega':  _try_float(raw.get('densBE_Omega'),  _DEFAULTS['densBE_Omega']),
    }


def _get_sim_folders(sweep_dir: str) -> dict:
    """Return {profile_tag: folder_path} for every sub-simulation."""
    sweep_path = Path(sweep_dir)
    sim_files = find_all_simulations(sweep_path)
    folders = {}
    for data_path in sim_files:
        tag = identify_profile_tag(data_path.parent.name)
        if tag and tag in PROFILE_STYLES:
            folders[tag] = data_path.parent
    return folders


# Density conversion: internal [1/pc³] -> CGS [cm⁻³]
NDENS_AU2CGS = INV_CONV.ndens_au2cgs


# =============================================================================
# Figure 1: Cloud Density & Enclosed Mass Profiles (STATIC, 2-panel)
# =============================================================================

def plot_enclosed_mass(sweep_dir: str, output_dir: Path, fmt: str = 'pdf',
                       show: bool = False) -> None:
    """
    Plot n(r) density profile and M_enc(r) for all four density profiles.

    Reads cloud parameters from ``_summary.txt`` in each simulation
    subfolder.  Falls back to hard-coded defaults when unavailable.

    Produces a single figure with two panels:
      (a) number-density  n(r)  [cm⁻³]  vs  r [pc]
      (b) enclosed mass   M(<r) [Msun]   vs  r [pc]
    """
    logger.info("Figure 1: Cloud Density & Enclosed Mass Profiles")

    # Discover per-run folders (for reading _summary.txt)
    sim_folders = _get_sim_folders(sweep_dir)

    # Profile configurations: (tag, profile_type, alpha_default, omega_default)
    profiles = [
        ('PL0',  'densPL', 0,    None),
        ('PL-1', 'densPL', -1,   None),
        ('PL-2', 'densPL', -2,   None),
        ('BE14', 'densBE', None, 14.1),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax_n, ax_M = axes

    for tag, ptype_default, alpha_default, omega_default in profiles:
        s = get_style(tag)

        # --- Read parameters from _summary.txt (or use defaults) ----------
        if tag in sim_folders:
            cp = get_cloud_params(sim_folders[tag])
        else:
            cp = dict(_DEFAULTS)

        mCloud  = cp['mCloud']
        nCore   = cp['nCore']          # 1/pc³
        rCore   = cp['rCore']          # pc
        nISM    = cp['nISM']           # 1/pc³
        mu_au   = cp['mu_ion']         # Msun
        rhoCore = nCore * mu_au        # Msun/pc³

        # Override profile type / shape params from summary when available
        ptype = cp['dens_profile'] if cp['dens_profile'] in ('densPL', 'densBE') else ptype_default
        alpha = cp['densPL_alpha'] if ptype == 'densPL' else alpha_default
        omega = cp['densBE_Omega'] if ptype == 'densBE' else omega_default

        # --- Compute n(r) and M(r) ----------------------------------------
        if ptype == 'densPL':
            # Cloud radius
            if alpha == 0:
                rCloud = compute_rCloud_homogeneous(mCloud, nCore, mu=mu_au)
            else:
                rCloud, _ = compute_rCloud_powerlaw(
                    mCloud, nCore, alpha, rCore=rCore, mu=mu_au
                )

            r_arr = np.logspace(np.log10(1e-3), np.log10(rCloud * 1.3), 500)
            n_arr = np.empty_like(r_arr)
            M_arr = np.empty_like(r_arr)

            if alpha == 0:
                inside = r_arr <= rCloud
                n_arr[inside]  = nCore
                n_arr[~inside] = nISM
                M_arr[inside]  = (4.0 / 3.0) * np.pi * r_arr[inside]**3 * rhoCore
                M_arr[~inside] = mCloud
            else:
                reg1 = r_arr <= rCore
                reg2 = (r_arr > rCore) & (r_arr <= rCloud)
                reg3 = r_arr > rCloud

                # density
                n_arr[reg1] = nCore
                n_arr[reg2] = nCore * (r_arr[reg2] / rCore) ** alpha
                n_arr[reg3] = nISM

                # enclosed mass
                M_arr[reg1] = (4.0 / 3.0) * np.pi * r_arr[reg1]**3 * rhoCore
                M_arr[reg2] = 4.0 * np.pi * rhoCore * (
                    rCore**3 / 3.0 +
                    (r_arr[reg2]**(3.0 + alpha) - rCore**(3.0 + alpha)) /
                    ((3.0 + alpha) * rCore**alpha)
                )
                M_arr[reg3] = mCloud

        elif ptype == 'densBE':
            le_sol = solve_lane_emden()
            be_result = create_BE_sphere(
                M_cloud=mCloud, n_core=nCore,
                Omega=omega, mu=mu_au
            )
            rCloud  = be_result.r_out
            xi_out  = be_result.xi_out
            m_dim_out = float(le_sol.f_m(xi_out))

            r_arr = np.logspace(np.log10(1e-3), np.log10(rCloud * 1.3), 500)
            n_arr = np.empty_like(r_arr)
            M_arr = np.empty_like(r_arr)

            inside = r_arr <= rCloud
            xi_inside = xi_out * (r_arr[inside] / rCloud)
            rho_ratio = le_sol.f_rho_rhoc(xi_inside)

            # density
            n_arr[inside]  = nCore * rho_ratio
            n_arr[~inside] = nISM

            # enclosed mass
            m_inside = le_sol.f_m(xi_inside)
            M_arr[inside]  = mCloud * (m_inside / m_dim_out)
            M_arr[~inside] = mCloud

        # --- Convert density to CGS for display --------------------------
        n_cgs = n_arr * NDENS_AU2CGS

        # --- Plot ---------------------------------------------------------
        ax_n.loglog(r_arr, n_cgs, color=s['color'], ls=s['ls'], lw=1.5,
                    label=s['label'])
        ax_M.loglog(r_arr, M_arr, color=s['color'], ls=s['ls'], lw=1.5,
                    label=s['label'])

    # Panel (a): density
    ax_n.set_xlabel(r'$r$ [pc]')
    ax_n.set_ylabel(r'$n(r)$ [cm$^{-3}$]')
    ax_n.set_title(r'(a) Density Profile')

    # Panel (b): enclosed mass
    ax_M.set_xlabel(r'$r$ [pc]')
    ax_M.set_ylabel(r'$M_{\rm enc}(<r)$ [M$_\odot$]')
    ax_M.set_title(r'(b) Enclosed Mass')

    add_legend(ax_n, PROFILE_ORDER, loc='best', fontsize=9)

    fig.tight_layout()
    savefig(fig, 'densityProfile_Menc', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Figure 2: Shell Evolution (3-panel)
# =============================================================================

def plot_shell_evolution(simulations: dict, output_dir: Path, fmt: str = 'pdf',
                         show: bool = False) -> None:
    """Plot R(t), v(t), M_shell(t) for all profiles.

    Includes phase-transition vertical lines (T = transition, M = momentum),
    horizontal rCloud lines on the radius panel, and rCloud-crossing markers,
    following the conventions in ``paper_radiusEvolution`` and ``plot_markers``.
    """
    logger.info("Figure 2: Shell Evolution")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    for tag in PROFILE_ORDER:
        if tag not in simulations:
            continue
        output = simulations[tag]
        s = get_style(tag)

        t = output.get('t_now')
        R2 = safe_get(output, 'R2')
        v2 = safe_get(output, 'v2') * V_AU2KMS  # convert pc/Myr -> km/s
        mshell = safe_get(output, 'shell_mass')

        # Phase array (for transition markers)
        phase = output.get('current_phase', as_array=False)
        if phase is not None:
            phase = np.asarray([str(p) for p in phase])
        # Collapse flag
        isCollapse_raw = output.get('isCollapse', as_array=False)
        isCollapse = None
        if isCollapse_raw is not None:
            isCollapse = np.array([bool(c) for c in isCollapse_raw])

        # Cloud radius (constant per run)
        rCloud = safe_get(output, 'rCloud')
        rCloud_val = rCloud[-1] if rCloud.size > 0 and rCloud[-1] > 0 else None

        # --- Add phase markers to all three panels (color-coded per profile) ---
        for ax in axes:
            add_plot_markers(
                ax, t,
                phase=phase,
                R2=R2 if ax is axes[0] else None,
                rcloud=rCloud_val if ax is axes[0] else None,
                isCollapse=isCollapse,
                dataset_color=s['color'],
                show_phase=True,
                show_rcloud=(ax is axes[0]),
                show_collapse=True,
                show_labels=True,
                show_momentum_labels=True,  # suppress M labels to reduce clutter
            )

        # Panel (a): R(t) with horizontal rCloud line
        axes[0].plot(t, R2, color=s['color'], ls=s['ls'], lw=1.5)
        if rCloud_val is not None:
            axes[0].axhline(rCloud_val, color=s['color'], ls='--',
                            lw=0.8, alpha=0.5)

        # Panel (b): v(t)
        axes[1].plot(t, v2, color=s['color'], ls=s['ls'], lw=1.5)

        # Panel (c): M_shell(t)
        axes[2].plot(t, mshell, color=s['color'], ls=s['ls'], lw=1.5)

    axes[0].set_xlabel(r'$t$ [Myr]')
    axes[0].set_ylabel(r'$R$ [pc]')
    axes[0].set_title(r'Shell Radius')
    axes[0].set_xlim(0,1)
    axes[0].set_ylim(0,5)

    axes[1].set_xlabel(r'$t$ [Myr]')
    axes[1].set_ylabel(r'$v$ [km\,s$^{-1}$]')
    axes[1].set_yscale('log')
    axes[1].set_title(r'Shell Velocity')

    axes[2].set_xlabel(r'$t$ [Myr]')
    axes[2].set_ylabel(r'$M_{\rm shell}$ [M$_\odot$]')
    axes[2].set_title(r'Shell Mass')

    # Legend: profile colours + marker entries
    marker_handles = get_marker_legend_handles(
        include_phase=True, include_rcloud=True, include_collapse=False
    )
    add_legend(axes[1], [t for t in PROFILE_ORDER if t in simulations],
               extra_handles=marker_handles, loc='best', fontsize=9)

    fig.tight_layout()
    savefig(fig, 'densityProfile_evolution', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Figure 3: Pressure Budget (2x2 panels)
# =============================================================================

def plot_pressure_budget(simulations: dict, output_dir: Path, fmt: str = 'pdf',
                         show: bool = False) -> None:
    """Plot pressure evolution for each profile in a 2x2 grid."""
    logger.info("Figure 3: Pressure Budget")

    tags_present = [t for t in PROFILE_ORDER if t in simulations]
    n = len(tags_present)
    nrows = 2
    ncols = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 6), squeeze=False,
                             sharey=True)

    for idx, tag in enumerate(tags_present):
        i, j = divmod(idx, ncols)
        ax = axes[i, j]
        output = simulations[tag]
        s = get_style(tag)

        t = output.get('t_now')
        Pb = safe_get(output, 'Pb')
        P_drive = safe_get(output, 'P_drive')
        P_IF = safe_get(output, 'P_IF')

        # Convert pressure from AU to P/k_B [K cm^-3]
        # Pb [Msun/Myr^2/pc] -> dyn/cm^2 -> P/k_B
        Pb_cgs = np.abs(Pb) * PB_AU2CGS
        P_drive_cgs = np.abs(P_drive) * PB_AU2CGS
        P_IF_cgs = np.abs(P_IF) * PB_AU2CGS

        Pb_over_kB = Pb_cgs / K_B_CGS
        P_drive_over_kB = P_drive_cgs / K_B_CGS
        P_IF_over_kB = P_IF_cgs / K_B_CGS

        # Compute radiation pressure from F_rad and R2
        R2 = safe_get(output, 'R2')
        F_rad = safe_get(output, 'F_rad')
        # P_rad = F_rad / (4 pi R^2), need to convert
        # F_rad is in AU force units, R2 is in pc
        R2_cm = R2 * INV_CONV.pc2cm
        F_rad_cgs = np.abs(F_rad) * F_AU2CGS
        with np.errstate(divide='ignore', invalid='ignore'):
            P_rad_cgs = np.where(R2_cm > 0,
                                 F_rad_cgs / (4.0 * np.pi * R2_cm**2), 0.0)
        P_rad_over_kB = P_rad_cgs / K_B_CGS

        # Plot
        mask_Pb = Pb_over_kB > 0
        mask_Pd = P_drive_over_kB > 0
        mask_IF = P_IF_over_kB > 0
        mask_Pr = P_rad_over_kB > 0

        if np.any(mask_Pb):
            ax.semilogy(t[mask_Pb], Pb_over_kB[mask_Pb],
                        color='#0072B2', ls='-', lw=1.2,
                        label=r'$P_{\rm b}$')
        if np.any(mask_Pd):
            ax.semilogy(t[mask_Pd], P_drive_over_kB[mask_Pd],
                        color='#D55E00', ls='--', lw=1.2,
                        label=r'$P_{\rm drive}$')
        if np.any(mask_IF):
            ax.semilogy(t[mask_IF], P_IF_over_kB[mask_IF],
                        color='#009E73', ls='-.', lw=1.2,
                        label=r'$P_{\rm IF}$')
        if np.any(mask_Pr):
            ax.semilogy(t[mask_Pr], P_rad_over_kB[mask_Pr],
                        color='#CC79A7', ls=':', lw=1.2,
                        label=r'$P_{\rm rad}$')

        ax.set_xlabel(r'$t$ [Myr]')
        ax.set_ylabel(r'$P/k_{\rm B}$ [K\,cm$^{-3}$]')
        ax.set_title(s['label'])
        ax.legend(fontsize=8, loc='best')

    # Turn off unused panels
    for idx in range(n, nrows * ncols):
        i, j = divmod(idx, ncols)
        axes[i, j].set_visible(False)

    fig.tight_layout()
    savefig(fig, 'densityProfile_pressure', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Figure 4: Force Budget (2x2 panels)
# =============================================================================

def plot_force_budget(simulations: dict, output_dir: Path, fmt: str = 'pdf',
                      show: bool = False) -> None:
    """Plot force evolution for each profile in a 2x2 grid."""
    logger.info("Figure 4: Force Budget")

    tags_present = [t for t in PROFILE_ORDER if t in simulations]
    n = len(tags_present)
    nrows = 2
    ncols = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 6), squeeze=False,
                             sharey=True)

    force_fields = [
        ('F_ram',     r'$F_{\rm ram}$',     '#0072B2', '-'),
        ('F_grav',    r'$|F_{\rm grav}|$',  '#D55E00', '--'),
        ('F_rad',     r'$F_{\rm rad}$',     '#009E73', '-.'),
        ('F_ion_out', r'$F_{\rm ion}$',     '#CC79A7', ':'),
    ]

    for idx, tag in enumerate(tags_present):
        i, j = divmod(idx, ncols)
        ax = axes[i, j]
        output = simulations[tag]
        s = get_style(tag)

        t = output.get('t_now')

        for field, label, color, ls in force_fields:
            F = safe_get(output, field)
            F_cgs = np.abs(F) * F_AU2CGS
            mask = F_cgs > 0
            if np.any(mask):
                ax.semilogy(t[mask], F_cgs[mask], color=color, ls=ls,
                            lw=1.2, label=label)

        ax.set_xlabel(r'$t$ [Myr]')
        ax.set_ylabel(r'$|F|$ [dyn]')
        ax.set_title(s['label'])
        ax.legend(fontsize=8, loc='best')

    for idx in range(n, nrows * ncols):
        i, j = divmod(idx, ncols)
        axes[i, j].set_visible(False)

    fig.tight_layout()
    savefig(fig, 'densityProfile_force', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Figure 5: Cumulative Momentum
# =============================================================================

def plot_cumulative_momentum(simulations: dict, output_dir: Path, fmt: str = 'pdf',
                             show: bool = False) -> None:
    """Plot cumulative momentum p = M_shell * v for all profiles."""
    logger.info("Figure 5: Cumulative Momentum")

    fig, ax = plt.subplots(figsize=(5, 4))

    for tag in PROFILE_ORDER:
        if tag not in simulations:
            continue
        output = simulations[tag]
        s = get_style(tag)

        t = output.get('t_now')
        v2 = safe_get(output, 'v2') * V_AU2KMS  # km/s
        mshell = safe_get(output, 'shell_mass')  # Msun

        # Momentum: M_shell * v [Msun km/s]
        p = mshell * v2

        ax.plot(t, np.abs(p), color=s['color'], ls=s['ls'], lw=1.5)

    ax.set_xlabel(r'$t$ [Myr]')
    ax.set_ylabel(r'$p = M_{\rm shell} \times v$ [M$_\odot$\,km\,s$^{-1}$]')
    ax.set_title(r'Cumulative Momentum')
    ax.set_yscale('log')

    add_legend(ax, [t for t in PROFILE_ORDER if t in simulations], loc='best')

    savefig(fig, 'densityProfile_momentum', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Figure 6: Phase Transition Timing
# =============================================================================

def plot_phase_timing(simulations: dict, output_dir: Path, fmt: str = 'pdf',
                      show: bool = False) -> None:
    """
    Plot phase transition timing as a horizontal stacked bar chart.

    Infers phase timing from the 'current_phase' field. If not available,
    falls back to Eb(t) analysis.
    """
    logger.info("Figure 6: Phase Transition Timing")

    fig, ax = plt.subplots(figsize=(6, 3))

    phase_colors = {
        'energy': '#0072B2',
        'implicit': '#0072B2',
        'transition': '#E69F00',
        'momentum': '#D55E00',
    }

    tags_present = [t for t in PROFILE_ORDER if t in simulations]
    y_positions = np.arange(len(tags_present))

    for yi, tag in enumerate(tags_present):
        output = simulations[tag]
        s = get_style(tag)

        t = output.get('t_now')
        phases = output.get('current_phase', as_array=False)

        # Determine phase intervals
        phase_intervals = []
        if phases is not None and len(phases) > 0:
            current_phase = str(phases[0])
            t_start = t[0]

            for ii in range(1, len(phases)):
                p = str(phases[ii])
                if p != current_phase:
                    phase_intervals.append((current_phase, t_start, t[ii]))
                    current_phase = p
                    t_start = t[ii]
            # Final interval
            phase_intervals.append((current_phase, t_start, t[-1]))

        # Plot as horizontal bars
        for phase_name, t0, t1 in phase_intervals:
            # Map phase names (energy/implicit -> energy phase)
            display_phase = phase_name
            if phase_name in ('energy', 'implicit'):
                display_phase = 'energy'
            color = phase_colors.get(display_phase, 'gray')
            ax.barh(yi, t1 - t0, left=t0, height=0.6, color=color,
                    edgecolor='white', lw=0.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([get_style(t)['label'] for t in tags_present])
    ax.set_xlabel(r'$t$ [Myr]')
    ax.set_title('Phase Timing')
    ax.set_xlim(0,1)

    # Legend
    legend_handles = [
        Patch(facecolor='#0072B2', label='Energy-driven'),
        Patch(facecolor='#E69F00', label='Transition'),
        Patch(facecolor='#D55E00', label='Momentum-driven'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9)

    fig.tight_layout()
    savefig(fig, 'densityProfile_phaseTime', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Figure 7: Blend Weight w(t)
# =============================================================================

def plot_blend_weight(simulations: dict, output_dir: Path, fmt: str = 'pdf',
                      show: bool = False) -> None:
    """Plot blending weight w_blend(t) for all profiles."""
    logger.info("Figure 7: Blend Weight")

    fig, ax = plt.subplots(figsize=(5, 4))

    has_data = False
    for tag in PROFILE_ORDER:
        if tag not in simulations:
            continue
        output = simulations[tag]
        s = get_style(tag)

        t = output.get('t_now')
        w = safe_get(output, 'w_blend')

        if np.any(w != 0):
            has_data = True
            ax.plot(t, w, color=s['color'], ls=s['ls'], lw=1.5)

    if not has_data:
        logger.warning("No w_blend data found in any simulation. Skipping Figure 7.")
        plt.close(fig)
        return

    ax.set_xlabel(r'$t$ [Myr]')
    ax.set_ylabel(r'$w_{\rm blend}$')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(r'HII Blending Weight')

    add_legend(ax, [t for t in PROFILE_ORDER if t in simulations], loc='best')

    savefig(fig, 'densityProfile_wblend', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Figure 8: Energy Retention
# =============================================================================

def plot_energy_retention(simulations: dict, output_dir: Path, fmt: str = 'pdf',
                          show: bool = False) -> None:
    """
    Plot energy retention E_b(t) / E_input(t) for all profiles.

    E_input = cumulative integral of Lmech_total over time.
    If Lmech_total is not available, plots E_b(t) alone.
    """
    logger.info("Figure 8: Energy Retention")

    fig, ax = plt.subplots(figsize=(5, 4))

    has_ratio = False
    for tag in PROFILE_ORDER:
        if tag not in simulations:
            continue
        output = simulations[tag]
        s = get_style(tag)

        t = output.get('t_now')
        Eb = safe_get(output, 'Eb')

        # Try to get cumulative wind energy input
        Lmech = safe_get(output, 'Lmech_total')

        if np.any(Lmech > 0):
            # Cumulative integral: E_input = integral of Lmech dt
            # Lmech is in [Msun pc^2 / Myr^3], t in [Myr]
            E_input = np.zeros_like(t)
            E_input[1:] = np.cumsum(0.5 * (Lmech[1:] + Lmech[:-1]) * np.diff(t))

            # Retention fraction
            with np.errstate(divide='ignore', invalid='ignore'):
                eta = np.where(E_input > 0, Eb / E_input, 0.0)

            mask = (E_input > 0) & np.isfinite(eta)
            if np.any(mask):
                has_ratio = True
                ax.plot(t[mask], eta[mask], color=s['color'], ls=s['ls'], lw=1.5)

    if has_ratio:
        ax.set_ylabel(r'$E_{\rm b} / E_{\rm input}$')
        ax.set_title(r'Energy Retention Fraction')
    else:
        # Fallback: just plot Eb(t)
        logger.warning("Lmech_total not available; plotting Eb(t) alone.")
        for tag in PROFILE_ORDER:
            if tag not in simulations:
                continue
            output = simulations[tag]
            s = get_style(tag)
            t = output.get('t_now')
            Eb = safe_get(output, 'Eb')
            mask = Eb > 0
            if np.any(mask):
                ax.semilogy(t[mask], Eb[mask], color=s['color'], ls=s['ls'],
                            lw=1.5)
        ax.set_ylabel(r'$E_{\rm b}$ [internal units]')
        ax.set_title(r'Bubble Energy')

    ax.set_xlabel(r'$t$ [Myr]')

    add_legend(ax, [t for t in PROFILE_ORDER if t in simulations], loc='best')

    savefig(fig, 'densityProfile_energyRetention', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Helper: map profile tags to data file paths (for external load_run calls)
# =============================================================================

def _get_profile_data_paths(sweep_dir: str) -> dict:
    """
    Get mapping from profile tag -> data file Path.

    Used by the grid figures that call load_run from external modules.
    """
    sweep_path = Path(sweep_dir)
    sim_files = find_all_simulations(sweep_path)
    paths = {}
    for data_path in sim_files:
        tag = identify_profile_tag(data_path.parent.name)
        if tag and tag in PROFILE_STYLES:
            paths[tag] = data_path
    return paths


# =============================================================================
# Figure 9: Feedback Fraction Grid (2x2)
# =============================================================================

def plot_feedback_grid(sweep_dir: str, output_dir: Path, fmt: str = 'pdf',
                       show: bool = False) -> None:
    """
    Plot feedback fraction stacked areas in a 2x2 grid.

    Each panel shows one density profile using the same stacked-area
    visualisation as paper_feedback.py.
    """
    import src._plots.paper_feedback as _fb
    from src._plots.plot_markers import get_marker_legend_handles

    logger.info("Figure 9: Feedback Fraction Grid")

    sim_paths = _get_profile_data_paths(sweep_dir)
    tags_present = [t for t in PROFILE_ORDER if t in sim_paths]

    if not tags_present:
        logger.warning("No simulations found for feedback grid. Skipping.")
        return

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 6), squeeze=False,
                             sharey=True)

    for idx, tag in enumerate(tags_present):
        i, j = divmod(idx, ncols)
        ax = axes[i, j]
        s = get_style(tag)

        try:
            t, R2, phase, base_forces, overlay_forces, rcloud, isCollapse = \
                _fb.load_run(sim_paths[tag])
            _fb.plot_run_on_ax(
                ax, t, R2, phase, base_forces, overlay_forces, rcloud,
                isCollapse, alpha=0.75, smooth_window=_fb.SMOOTH_WINDOW,
                phase_change=_fb.PHASE_CHANGE, use_log_x=_fb.USE_LOG_X,
            )
        except Exception as e:
            logger.error(f"Feedback grid {tag}: {e}")
            ax.text(0.5, 0.5, "error", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue

        ax.set_title(s['label'])
        if i == nrows - 1:
            ax.set_xlabel(r'$t$ [Myr]')
        if j == 0:
            ax.set_ylabel(r'$F/F_{\rm tot}$')
        else:
            ax.tick_params(labelleft=False)

    for idx in range(len(tags_present), nrows * ncols):
        i, j = divmod(idx, ncols)
        axes[i, j].set_visible(False)

    # Global legend
    handles = []
    for field, label, color in _fb.FORCE_FIELDS_BASE:
        ec = '0.4' if color == 'white' else 'none'
        a = 1.0 if color == 'white' else 0.75
        handles.append(Patch(facecolor=color, edgecolor=ec, alpha=a,
                             label=label))
    if _fb.INCLUDE_ALL_FORCE:
        handles.append(Patch(facecolor='none', edgecolor=_fb.C_RAM,
                             hatch='////', label='Wind'))
        handles.append(Patch(facecolor='none', edgecolor=_fb.C_SN,
                             hatch='\\\\\\\\', label='SN'))
    handles.extend(get_marker_legend_handles())

    fig.legend(handles=handles, loc='upper center', ncol=4,
               frameon=True, facecolor='white', framealpha=0.9,
               edgecolor='0.2', bbox_to_anchor=(0.5, 1.05), fontsize=8)
    fig.subplots_adjust(top=0.88)

    savefig(fig, 'densityProfile_feedback', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Figure 10: Momentum Grid (2x2)
# =============================================================================

def plot_momentum_grid(sweep_dir: str, output_dir: Path, fmt: str = 'pdf',
                       show: bool = False) -> None:
    """
    Plot cumulative momentum lines in a 2x2 grid.

    Each panel shows one density profile using the same momentum-line
    visualisation as paper_momentum.py.
    """
    import src._plots.paper_momentum as _mom
    from src._plots.plot_markers import get_marker_legend_handles

    logger.info("Figure 10: Momentum Grid")

    sim_paths = _get_profile_data_paths(sweep_dir)
    tags_present = [t for t in PROFILE_ORDER if t in sim_paths]

    if not tags_present:
        logger.warning("No simulations found for momentum grid. Skipping.")
        return

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 6), squeeze=False,
                             sharey=True)

    for idx, tag in enumerate(tags_present):
        i, j = divmod(idx, ncols)
        ax = axes[i, j]
        s = get_style(tag)

        try:
            t, r, phase, forces, forces_dict, rcloud, isCollapse = \
                _mom.load_run(sim_paths[tag])
            _mom.plot_momentum_lines_on_ax(
                ax, t, r, phase, forces, forces_dict, rcloud, isCollapse,
                smooth_window=_mom.SMOOTH_WINDOW, phase_change=_mom.PHASE_CHANGE,
            )
        except Exception as e:
            logger.error(f"Momentum grid {tag}: {e}")
            ax.text(0.5, 0.5, "error", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue

        ax.set_title(s['label'])
        if i == nrows - 1:
            ax.set_xlabel(r'$t$ [Myr]')
        if j == 0:
            ax.set_ylabel(r'$p(t) = \int F\,dt$')

    for idx in range(len(tags_present), nrows * ncols):
        i, j = divmod(idx, ncols)
        axes[i, j].set_visible(False)

    # Global legend
    handles = []
    for _, label, color in _mom.FORCE_FIELDS:
        handles.append(Line2D([0], [0], color=color, lw=1.6, ls='-',
                              label=label))
    for _, label, color in _mom.DASHED_FIELDS:
        handles.append(Line2D([0], [0], color=color, lw=1.2, ls='--',
                              label=label))
    handles.append(Line2D([0], [0], color='darkgrey', lw=2.4, label='Net'))
    handles.extend(get_marker_legend_handles())

    fig.legend(handles=handles, loc='upper center', ncol=4,
               frameon=True, facecolor='white', framealpha=0.9,
               edgecolor='0.2', bbox_to_anchor=(0.5, 1.05), fontsize=8)
    fig.subplots_adjust(top=0.88)

    savefig(fig, 'densityProfile_momentum', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Figure 11: Thermal Regime Grid (2x2)
# =============================================================================

def plot_thermal_grid(sweep_dir: str, output_dir: Path, fmt: str = 'pdf',
                      show: bool = False) -> None:
    """
    Plot thermal regime (w_blend) in a 2x2 grid.

    Each panel shows one density profile using the same thermal-regime
    visualisation as paper_thermalRegime.py.
    """
    import src._plots.paper_thermalRegime as _therm
    from src._plots.plot_markers import get_marker_legend_handles

    logger.info("Figure 11: Thermal Regime Grid")

    sim_paths = _get_profile_data_paths(sweep_dir)
    tags_present = [t for t in PROFILE_ORDER if t in sim_paths]

    if not tags_present:
        logger.warning("No simulations found for thermal grid. Skipping.")
        return

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 6), squeeze=False,
                             sharey=True)

    for idx, tag in enumerate(tags_present):
        i, j = divmod(idx, ncols)
        ax = axes[i, j]
        s = get_style(tag)

        try:
            data = _therm.load_run(sim_paths[tag])
            _therm.plot_run_on_ax(
                ax, data, smooth_window=_therm.SMOOTH_WINDOW,
                phase_change=_therm.PHASE_CHANGE,
                plot_mode=_therm.PLOT_MODE, use_log_x=_therm.USE_LOG_X,
            )
        except Exception as e:
            logger.error(f"Thermal grid {tag}: {e}")
            ax.text(0.5, 0.5, "error", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue

        ax.set_title(s['label'])
        if i == nrows - 1:
            ax.set_xlabel(r'$t$ [Myr]')
        if j == 0:
            ax.set_ylabel(r'$w_{\rm blend}$')
        else:
            ax.tick_params(labelleft=False)

    for idx in range(len(tags_present), nrows * ncols):
        i, j = divmod(idx, ncols)
        axes[i, j].set_visible(False)

    # Global legend
    if _therm.PLOT_MODE == 'stacked':
        handles = [
            Patch(facecolor=_therm.C_BUBBLE, alpha=0.7,
                  label=r'$(1-w)$ Bubble'),
            Patch(facecolor=_therm.C_HII, alpha=0.7, label=r'$w$ HII'),
        ]
    else:
        handles = [
            Line2D([0], [0], color='black', lw=2, label=r'$w_{\rm blend}$'),
            Patch(facecolor=_therm.C_BUBBLE, alpha=0.1, edgecolor='none',
                  label='Bubble regime'),
            Patch(facecolor=_therm.C_HII, alpha=0.1, edgecolor='none',
                  label='HII regime'),
        ]
    handles.extend(get_marker_legend_handles())

    fig.legend(handles=handles, loc='upper center', ncol=4,
               frameon=True, facecolor='white', framealpha=0.9,
               edgecolor='0.2', bbox_to_anchor=(0.5, 1.05), fontsize=8)
    fig.subplots_adjust(top=0.88)

    savefig(fig, 'densityProfile_thermal', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Main entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TRINITY Density Profile Comparison Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_densityProfile.py -F outputs/density_profile_sweep
  python paper_densityProfile.py -F outputs/density_profile_sweep --fmt png
  python paper_densityProfile.py -F outputs/density_profile_sweep --show
  python paper_densityProfile.py -F outputs/density_profile_sweep -o fig/sweep/
        """
    )
    parser.add_argument(
        '--folder', '-F', required=True,
        help='Path to density profile sweep output directory (required)'
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help='Directory to save figures (default: fig/density_profile_sweep/)'
    )
    parser.add_argument(
        '--fmt', default='pdf',
        help='Figure format (default: pdf)'
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Show figures interactively instead of just saving'
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        folder_name = Path(args.folder).name
        output_dir = FIG_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Enclosed mass (static, no simulation data needed)
    try:
        plot_enclosed_mass(args.folder, output_dir, args.fmt, args.show)
    except Exception as e:
        logger.error(f"Figure 1 (Enclosed Mass) failed: {e}")

    # Load simulation data for remaining figures
    try:
        simulations = load_sweep_simulations(args.folder)
    except FileNotFoundError as e:
        logger.error(f"Could not load simulations: {e}")
        logger.info("Only Figure 1 (static enclosed mass) was generated.")
        return

    if not simulations:
        logger.error("No valid simulations found. Exiting.")
        return

    # Generate all simulation-based figures
    plot_functions = [
        ("Figure 2: Shell Evolution",      plot_shell_evolution),
        ("Figure 3: Pressure Budget",       plot_pressure_budget),
        ("Figure 4: Force Budget",          plot_force_budget),
        ("Figure 5: Cumulative Momentum",   plot_cumulative_momentum),
        ("Figure 6: Phase Timing",          plot_phase_timing),
        ("Figure 7: Blend Weight",          plot_blend_weight),
        ("Figure 8: Energy Retention",      plot_energy_retention),
    ]

    for name, func in plot_functions:
        try:
            func(simulations, output_dir, args.fmt, args.show)
        except Exception as e:
            logger.error(f"{name} failed: {e}")

    # Grid figures (reuse plot functions from paper_feedback/momentum/thermal)
    grid_functions = [
        ("Figure 9: Feedback Grid",   plot_feedback_grid),
        ("Figure 10: Momentum Grid",  plot_momentum_grid),
        ("Figure 11: Thermal Grid",   plot_thermal_grid),
    ]

    for name, func in grid_functions:
        try:
            func(args.folder, output_dir, args.fmt, args.show)
        except Exception as e:
            logger.error(f"{name} failed: {e}")

    logger.info(f"All figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
