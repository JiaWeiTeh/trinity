#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Density Profile Comparison Diagnostics for TRINITY.

Compares four density profiles (same cloud mass, SFE, core density, but varying
density structure) from a density_profile_sweep run:
  - Power-law rho ~ r^0  (uniform)
  - Power-law rho ~ r^-1
  - Power-law rho ~ r^-2
  - Critical Bonnor-Ebert sphere

Produces 8 diagnostic figures (1 static + 7 simulation-based) examining how
cloud density structure affects feedback-driven shell evolution.

Usage:
  python paper_densityProfile.py -F <path_to_density_profile_sweep_output>
  python paper_densityProfile.py -F outputs/density_profile_sweep --fmt png
  python paper_densityProfile.py -F outputs/density_profile_sweep --show

@author: Jia Wei Teh
"""

import re
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from src._plots.plot_base import FIG_DIR
from src._output.trinity_reader import (
    load_output, find_all_simulations, TrinityOutput
)
from src._plots.grid_template import filter_sim_files_by_phii
from src._functions.unit_conversions import CONV, INV_CONV, CGS
from src.cloud_properties.bonnorEbertSphere import (
    solve_lane_emden, create_BE_sphere
)
from src.cloud_properties.powerLawSphere import (
    compute_rCloud_homogeneous, compute_rCloud_powerlaw
)
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles

# =============================================================================
# MARKER DEFAULTS (off for clean paper figures; enable via CLI --show-*)
# =============================================================================
SHOW_PHASE = False
SHOW_RCLOUD = False
SHOW_RCLOUD_H = False
SHOW_COLLAPSE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Global matplotlib style (matches paper_ODIN)
# =============================================================================
plt.rcParams.update({
    'font.size':        20,
    'axes.labelsize':   20,
    'axes.titlesize':   20,
    'xtick.labelsize':  18,
    'ytick.labelsize':  18,
    'legend.fontsize':  16,
})

# =============================================================================
# Constants
# =============================================================================

# Colourblind-safe palette (Wong 2011) with solid lines for clarity
PROFILE_STYLES = {
    'PL0':  {'color': '#0072B2', 'ls': '-',  'label': r'$\rho \propto r^{0}$'},
    'PL-1': {'color': '#D55E00', 'ls': '-',  'label': r'$\rho \propto r^{-1}$'},
    'PL-2': {'color': '#009E73', 'ls': '-',  'label': r'$\rho \propto r^{-2}$'},
    'BE14': {'color': '#CC79A7', 'ls': '-',  'label': 'Bonnor-Ebert'},
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
    sim_files = filter_sim_files_by_phii(sim_files, "yes")

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
                              label=s['label'].replace('\n', ' ')))
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
    sim_files = filter_sim_files_by_phii(sim_files, "yes")
    folders = {}
    for data_path in sim_files:
        tag = identify_profile_tag(data_path.parent.name)
        if tag and tag in PROFILE_STYLES:
            folders[tag] = data_path.parent
    return folders


# Density conversion: internal [1/pc³] -> CGS [cm⁻³]
NDENS_AU2CGS = INV_CONV.ndens_au2cgs


# =============================================================================
# Density profile ingredients helper (used by Fig. 1 and Fig. 2 top panel)
# =============================================================================

# Per-tag defaults: (profile_type, alpha_default, omega_default)
_PROFILE_DEFAULTS = {
    'PL0':  ('densPL', 0,    None),
    'PL-1': ('densPL', -1,   None),
    'PL-2': ('densPL', -2,   None),
    'BE14': ('densBE', None, 14.1),
}


def _compute_rho_M_profile(tag: str, sim_folders: dict):
    """Return (r_arr [pc], n_cgs [cm^-3], M_arr [Msun]) for profile *tag*."""
    ptype_default, alpha_default, omega_default = _PROFILE_DEFAULTS[tag]

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

    ptype = cp['dens_profile'] if cp['dens_profile'] in ('densPL', 'densBE') else ptype_default
    alpha = cp['densPL_alpha'] if ptype == 'densPL' else alpha_default
    omega = cp['densBE_Omega'] if ptype == 'densBE' else omega_default

    if ptype == 'densPL':
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

            n_arr[reg1] = nCore
            n_arr[reg2] = nCore * (r_arr[reg2] / rCore) ** alpha
            n_arr[reg3] = nISM

            M_arr[reg1] = (4.0 / 3.0) * np.pi * r_arr[reg1]**3 * rhoCore
            M_arr[reg2] = 4.0 * np.pi * rhoCore * (
                rCore**3 / 3.0 +
                (r_arr[reg2]**(3.0 + alpha) - rCore**(3.0 + alpha)) /
                ((3.0 + alpha) * rCore**alpha)
            )
            M_arr[reg3] = mCloud

    else:  # densBE
        le_sol = solve_lane_emden()
        be_result = create_BE_sphere(
            M_cloud=mCloud, n_core=nCore,
            Omega=omega, mu=mu_au
        )
        rCloud    = be_result.r_out
        xi_out    = be_result.xi_out
        m_dim_out = float(le_sol.f_m(xi_out))

        r_arr = np.logspace(np.log10(1e-3), np.log10(rCloud * 1.3), 500)
        n_arr = np.empty_like(r_arr)
        M_arr = np.empty_like(r_arr)

        inside = r_arr <= rCloud
        xi_inside = xi_out * (r_arr[inside] / rCloud)
        rho_ratio = le_sol.f_rho_rhoc(xi_inside)

        n_arr[inside]  = nCore * rho_ratio
        n_arr[~inside] = nISM

        m_inside = le_sol.f_m(xi_inside)
        M_arr[inside]  = mCloud * (m_inside / m_dim_out)
        M_arr[~inside] = mCloud

    return r_arr, n_arr * NDENS_AU2CGS, M_arr


def _extend_outer_plateau(r_arr, n_cgs, M_arr, r_max):
    """Append a point at *r_max* so the nISM (and M_cloud) plateau reaches it.

    Different profiles have different rCloud values, so each per-tag r_arr
    ends at rCloud*1.3. When the curves are overlaid on a shared x-axis,
    profiles with smaller rCloud would otherwise terminate before the
    right-hand edge of the plot — extend their final (nISM / M_cloud) value
    out to the common r_max so the horizontal nISM line is visible all the
    way across.
    """
    if r_arr[-1] >= r_max:
        return r_arr, n_cgs, M_arr
    r_arr = np.append(r_arr, r_max)
    n_cgs = np.append(n_cgs, n_cgs[-1])
    M_arr = np.append(M_arr, M_arr[-1])
    return r_arr, n_cgs, M_arr


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

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax_n, ax_M = axes

    # Pre-compute every profile so we know the global outer radius, then
    # extend each curve's nISM plateau out to that radius before plotting
    # — this keeps the nISM horizontal line visible all the way to the
    # right-hand edge of the panel even for profiles with smaller rCloud.
    profiles = {tag: _compute_rho_M_profile(tag, sim_folders)
                for tag in PROFILE_ORDER}
    r_max = max(r_arr[-1] for r_arr, _, _ in profiles.values())

    for tag in PROFILE_ORDER:
        s = get_style(tag)
        r_arr, n_cgs, M_arr = _extend_outer_plateau(*profiles[tag], r_max)

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

    add_legend(ax_n, PROFILE_ORDER, loc='best')

    fig.tight_layout()
    savefig(fig, 'densityProfile_Menc', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Figure 2: Shell Evolution (3-panel)
# =============================================================================

# Alpha for the dashed M_enc line on the top panel (shared by evolution plots).
_MENC_ALPHA = 0.45


def _setup_time_panel_ticks(ax):
    """Apply consistent inward major/minor ticks on all four sides."""
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, right=True)
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', length=3)


def _draw_ingredients_panel(ax_rho, ax_M, tags_present: list,
                            sim_folders: dict) -> None:
    """Draw the density profile (solid) + M_enc (dashed, twinx) top panel.

    Configures ticks, scales, axis labels and the in-panel solid/dashed
    legend. y-axes are plotted as log10 values on a linear scale so that
    minor ticks are legible across many decades; x stays on a log scale.
    """
    # Ticks: ax_rho owns the left, ax_M owns the right
    ax_rho.minorticks_on()
    ax_rho.tick_params(axis='x', which='both', direction='in', top=True)
    ax_rho.tick_params(axis='y', which='both', direction='in',
                       left=True, right=False)
    ax_rho.tick_params(which='major', length=5)
    ax_rho.tick_params(which='minor', length=3)

    ax_M.minorticks_on()
    ax_M.tick_params(axis='x', which='both', top=False, bottom=False,
                     labeltop=False, labelbottom=False)
    ax_M.tick_params(axis='y', which='both', direction='in',
                     right=True, left=False, labelleft=False)
    ax_M.tick_params(which='major', length=5)
    ax_M.tick_params(which='minor', length=3)

    # Pre-compute every profile so we know the global outer radius, then
    # extend each curve's nISM plateau out to that radius before plotting
    # — this keeps the nISM horizontal line visible all the way to the
    # right-hand edge of the panel even for profiles with smaller rCloud.
    profiles = {}
    for tag in tags_present:
        try:
            profiles[tag] = _compute_rho_M_profile(tag, sim_folders)
        except Exception as e:
            logger.warning(f"Could not compute profile ingredients for {tag}: {e}")
    if not profiles:
        return
    r_max = max(r_arr[-1] for r_arr, _, _ in profiles.values())

    for tag in tags_present:
        if tag not in profiles:
            continue
        s = get_style(tag)
        r_arr, n_cgs, M_arr = _extend_outer_plateau(*profiles[tag], r_max)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_n = np.log10(n_cgs)
            log_M = np.log10(M_arr)
        ax_rho.plot(r_arr, log_n, color=s['color'], ls='-', lw=1.5)
        ax_M.plot(r_arr, log_M, color=s['color'], ls='--', lw=1.2,
                  alpha=_MENC_ALPHA)

    ax_rho.set_xscale('log')
    ax_M.set_xscale('log')

    ax_rho.set_xlabel(r'$r$ [pc]')
    ax_rho.set_ylabel(r'$\log_{10}\!\left(n_{\rm cloud}(r) / {\rm cm}^{-3}\right)$')
    # Twiny label: rotated the other way (reading top-to-bottom) to match
    # the right-hand side of the panel.
    ax_M.set_ylabel(
        r'$\log_{10}\!\left(M_{\rm enc}(<r) / {\rm M}_\odot\right)$',
        rotation=270, labelpad=22, va='bottom',
    )

    style_handles = [
        Line2D([0], [0], color='black', ls='-',  lw=1.5,
               label=r'$n_{\rm cloud}(r)$'),
        Line2D([0], [0], color='black', ls='--', lw=1.2, alpha=_MENC_ALPHA,
               label=r'$M_{\rm enc}(<r)$'),
    ]
    ax_rho.legend(handles=style_handles, loc='upper left', frameon=False,
                  handlelength=2.0, handletextpad=0.5)


def _draw_Rb_panel(ax, simulations: dict, tags_present: list) -> None:
    """Draw the R_b(t) panel (outer bubble radius) with phase markers."""
    _setup_time_panel_ticks(ax)
    for tag in tags_present:
        output = simulations[tag]
        s = get_style(tag)

        t = output.get('t_now')
        R2 = safe_get(output, 'R2')

        phase_raw = output.get('current_phase', as_array=False)
        phase = (np.asarray([str(p) for p in phase_raw])
                 if phase_raw is not None else None)
        isCollapse_raw = output.get('isCollapse', as_array=False)
        isCollapse = (np.array([bool(c) for c in isCollapse_raw])
                      if isCollapse_raw is not None else None)
        rCloud = safe_get(output, 'rCloud')
        rCloud_val = rCloud[-1] if rCloud.size > 0 and rCloud[-1] > 0 else None

        add_plot_markers(
            ax, t,
            phase=phase,
            R2=R2,
            rcloud=rCloud_val,
            isCollapse=isCollapse,
            dataset_color=s['color'],
            show_phase=SHOW_PHASE,
            show_rcloud=SHOW_RCLOUD,
            show_rcloud_horizontal=SHOW_RCLOUD_H,
            show_collapse=SHOW_COLLAPSE,
            show_labels=True,
        )
        ax.plot(t, R2, color=s['color'], ls=s['ls'], lw=1.5)
        if rCloud_val is not None and SHOW_RCLOUD_H:
            ax.axhline(rCloud_val, color=s['color'], ls='--',
                       lw=0.8, alpha=0.5)

    ax.set_ylabel(r'$R_{\rm b}$ [pc]')


def _build_profile_figure_legend(fig, tags_present: list,
                                 legend_y: float = 1.0):
    """Figure-level 2x2 legend of profile colours with a border.

    ``legend_y`` is the figure-fraction *top* position for the legend box
    (``loc='upper center'``). Callers should compute this via
    :func:`_compute_legend_layout` from ``grid_template``.
    """
    profile_handles = [
        Line2D([0], [0], color=get_style(tag)['color'], ls='-', lw=1.8,
               label=get_style(tag)['label'].replace('\n', ' '))
        for tag in tags_present
    ]
    marker_handles = list(get_marker_legend_handles(
        include_phase=False,
        include_rcloud=SHOW_RCLOUD,
        include_rcloud_horizontal=SHOW_RCLOUD_H,
        include_collapse=False,
    ))
    fig.legend(
        handles=profile_handles + marker_handles,
        loc='upper center',
        ncol=2,
        bbox_to_anchor=(0.5, legend_y),
        frameon=True,
        facecolor='white',
        edgecolor='0.2',
        framealpha=0.95,
        columnspacing=1.5,
        handletextpad=0.5,
    )


def plot_shell_evolution(simulations: dict, output_dir: Path, fmt: str = 'pdf',
                         show: bool = False, sweep_dir: str = None) -> None:
    """Plot stacked density profile (+M_enc) and R(t), v(t), M_shell(t).

    Top row (row 0) shows the density profile as solid lines and the enclosed
    mass as dashed lines on a twiny axis (x-axis is radius, independent).
    Rows 1-3 share the same time x-axis and show shell radius, velocity, and
    mass respectively. Legend is at the top as a figure-level legend.
    """
    from src._plots.grid_template import _compute_legend_layout

    logger.info("Figure 2: Shell Evolution (stacked)")

    tags_present = [tag for tag in PROFILE_ORDER if tag in simulations]

    # Two independent gridspecs so that the top "ingredients" panel (with its
    # own r-axis) is clearly separated from the three time-evolution panels
    # that share a common t-axis.
    fig_h = 13.0
    fig = plt.figure(figsize=(6.5, fig_h))

    layout = _compute_legend_layout(
        fig_h, n_legend_items=len(tags_present), legend_ncol=2,
    )

    # The ingredient panel occupies the top strip; the three time-panels
    # fill the rest below.  A gap separates them so the ingredient-panel
    # xlabel is never overlapped by the R_b panel below.
    top_of_top_panel = layout['top']
    bot_of_top_panel = top_of_top_panel - 0.20
    gap = 0.05
    top_of_time_panels = bot_of_top_panel - gap
    gs_top = fig.add_gridspec(
        1, 1,
        top=top_of_top_panel, bottom=bot_of_top_panel,
        left=0.15, right=0.87,
    )
    gs_bot = fig.add_gridspec(
        3, 1,
        top=top_of_time_panels, bottom=0.05, hspace=0.08,
        left=0.15, right=0.87,
    )

    ax_rho = fig.add_subplot(gs_top[0, 0])
    ax_M   = ax_rho.twinx()                               # enclosed mass twiny
    ax_R   = fig.add_subplot(gs_bot[0, 0])
    ax_v   = fig.add_subplot(gs_bot[1, 0], sharex=ax_R)
    ax_m   = fig.add_subplot(gs_bot[2, 0], sharex=ax_R)

    # Ticks for the three time-evolution panels
    for ax in (ax_R, ax_v, ax_m):
        _setup_time_panel_ticks(ax)

    # --- Row 0: density + M_enc (with ticks, labels, legend) ---
    sim_folders = _get_sim_folders(sweep_dir) if sweep_dir else {}
    _draw_ingredients_panel(ax_rho, ax_M, tags_present, sim_folders)

    # --- Rows 1-3: time evolution, shared x-axis ---
    # The R_b panel is drawn via the shared helper; v and M_shell inline.
    _draw_Rb_panel(ax_R, simulations, tags_present)

    for tag in tags_present:
        output = simulations[tag]
        s = get_style(tag)

        t = output.get('t_now')
        v2 = safe_get(output, 'v2') * V_AU2KMS  # pc/Myr -> km/s
        mshell = safe_get(output, 'shell_mass')

        phase_raw = output.get('current_phase', as_array=False)
        phase = (np.asarray([str(p) for p in phase_raw])
                 if phase_raw is not None else None)
        isCollapse_raw = output.get('isCollapse', as_array=False)
        isCollapse = (np.array([bool(c) for c in isCollapse_raw])
                      if isCollapse_raw is not None else None)

        for ax in (ax_v, ax_m):
            add_plot_markers(
                ax, t,
                phase=phase,
                isCollapse=isCollapse,
                dataset_color=s['color'],
                show_phase=SHOW_PHASE,
                show_collapse=SHOW_COLLAPSE,
                show_labels=True,
            )

        ax_v.plot(t, v2,     color=s['color'], ls=s['ls'], lw=1.5)
        ax_m.plot(t, mshell, color=s['color'], ls=s['ls'], lw=1.5)

    ax_v.set_ylabel(r'$v$ [km\,s$^{-1}$]')
    ax_v.set_yscale('log')
    ax_m.set_ylabel(r'$M_{\rm shell}$ [M$_\odot$]')
    ax_m.set_xlabel(r'$t$ [Myr]')

    # Hide tick labels (not ticks themselves) on shared x-axis for rows 1-2
    plt.setp(ax_R.get_xticklabels(), visible=False)
    plt.setp(ax_v.get_xticklabels(), visible=False)

    _build_profile_figure_legend(fig, tags_present,
                                 legend_y=layout['legend_y'])

    savefig(fig, 'densityProfile_evolution', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Figure 2 (paper version): top ingredients panel + R_b(t) only
# =============================================================================

def plot_shell_evolution_paper(simulations: dict, output_dir: Path,
                               fmt: str = 'pdf', show: bool = False,
                               sweep_dir: str = None) -> None:
    """Paper-ready 2-panel version of :func:`plot_shell_evolution`.

    Shows only the top (density + M_enc ingredients) panel and the R_b(t)
    panel below it. Legend-spacing follows the ``paper_feedback`` convention
    via :func:`_compute_legend_layout` so the legend never overlaps the
    top subplot; a generous ``hspace`` keeps the ingredient-panel xlabel
    clear of the R_b panel below.
    """
    from src._plots.grid_template import _compute_legend_layout

    logger.info("Figure 2p: Shell Evolution (paper, 2-panel)")

    tags_present = [tag for tag in PROFILE_ORDER if tag in simulations]

    fig_h = 7.0
    fig = plt.figure(figsize=(7, fig_h))

    # Compute legend + axes vertical layout so the legend doesn't clip
    # the top panel.  Legend has 2 rows (4 handles, ncol=2).
    layout = _compute_legend_layout(
        fig_h, n_legend_items=len(tags_present), legend_ncol=2,
    )

    # Two stacked panels with a wide hspace so the top panel's xlabel
    # (r [pc]) stays clear of the R_b(t) panel below.
    gs = fig.add_gridspec(
        2, 1,
        hspace=0.45,
        left=0.14, right=0.86,
        top=layout['top'], bottom=0.10,
    )
    ax_rho = fig.add_subplot(gs[0, 0])
    ax_M   = ax_rho.twinx()
    ax_R   = fig.add_subplot(gs[1, 0])

    sim_folders = _get_sim_folders(sweep_dir) if sweep_dir else {}
    _draw_ingredients_panel(ax_rho, ax_M, tags_present, sim_folders)
    _draw_Rb_panel(ax_R, simulations, tags_present)

    ax_R.set_xlabel(r'$t$ [Myr]')

    _build_profile_figure_legend(fig, tags_present,
                                 legend_y=layout['legend_y'])

    savefig(fig, 'densityProfile_paper', output_dir, fmt)
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

    tags_present = [tag for tag in PROFILE_ORDER if tag in simulations]
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
        ax.set_title(s['label'].replace('\n', ' '))
        ax.legend(loc='best')

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

    tags_present = [tag for tag in PROFILE_ORDER if tag in simulations]
    n = len(tags_present)
    nrows = 2
    ncols = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 6), squeeze=False,
                             sharey=True)

    force_fields = [
        ('F_ram',     r'$F_{\rm ram}$',     '#0072B2', '-'),
        ('F_grav',    r'$|F_{\rm grav}|$',  '#D55E00', '--'),
        ('F_rad',     r'$F_{\rm rad}$',     '#009E73', '-.'),
        ('F_HII',     r'$F_{\rm HII}$',      '#CC79A7', ':'),
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
        ax.set_title(s['label'].replace('\n', ' '))
        ax.legend(loc='best')

    for idx in range(n, nrows * ncols):
        i, j = divmod(idx, ncols)
        axes[i, j].set_visible(False)

    fig.tight_layout()
    savefig(fig, 'densityProfile_force', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Figure 5: Phase Transition Timing
# =============================================================================

def plot_phase_timing(simulations: dict, output_dir: Path, fmt: str = 'pdf',
                      show: bool = False) -> None:
    """Backward-compatible wrapper — calls plot_phase_timeline."""
    plot_phase_timeline(simulations, output_dir, fmt, show)


def _extract_phase_info(output: TrinityOutput) -> dict:
    """
    Extract phase intervals and key event times from a simulation.

    Returns
    -------
    dict with keys:
        intervals : list of (phase_label, t_start, t_end)
            Phase labels: 'energy', 'transition', 'momentum', 'collapse'
        t_trans : float or None
            Time when transition phase begins.
        t_turn : float or None
            Shell turnaround time (v2 crosses zero from positive to negative).
        t_Rcloud : float or None
            Time when R2 >= rCloud.
        t_end : float
            Last timestep.
        outcome : str
            'expanding' or 're-collapse'
    """
    t = output.get('t_now')
    phases = np.array(output.get('current_phase', as_array=False))
    v2 = safe_get(output, 'v2')
    R2 = safe_get(output, 'R2')
    isCollapse = np.array(output.get('isCollapse', as_array=False))
    rCloud = float(output[0].get('rCloud', np.nan))

    # --- Phase intervals ---
    # Map energy/implicit -> 'energy' for display
    def map_phase(p):
        p = str(p)
        if p in ('energy', 'implicit'):
            return 'energy'
        return p

    mapped = [map_phase(p) for p in phases]

    # Build raw intervals
    raw_intervals = []
    current = mapped[0]
    t_start = t[0]
    for ii in range(1, len(mapped)):
        if mapped[ii] != current:
            raw_intervals.append((current, t_start, t[ii]))
            current = mapped[ii]
            t_start = t[ii]
    raw_intervals.append((current, t_start, t[-1]))

    # Detect collapse: find turnaround time where v2 crosses zero
    # (positive -> negative) after having been positive
    t_turn = None
    pos_mask = v2 > 0
    if np.any(pos_mask):
        # Find first index where v2 goes from positive to non-positive
        for ii in range(1, len(v2)):
            if v2[ii - 1] > 0 and v2[ii] <= 0:
                # Linear interpolation for exact crossing
                if v2[ii - 1] != v2[ii]:
                    t_turn = t[ii - 1] + (0 - v2[ii - 1]) * (t[ii] - t[ii - 1]) / (v2[ii] - v2[ii - 1])
                else:
                    t_turn = t[ii]
                break

    # Determine outcome
    is_collapse = False
    if isCollapse is not None:
        # Check if any isCollapse is True
        for val in isCollapse:
            if val is True or val == 'True' or val == 1:
                is_collapse = True
                break
    if t_turn is not None and v2[-1] < 0:
        is_collapse = True
    outcome = 're-collapse' if is_collapse else 'expanding'

    # Split final momentum interval at t_turn to show collapse phase
    intervals = []
    for phase_name, t0, t1 in raw_intervals:
        if phase_name == 'momentum' and t_turn is not None and t0 < t_turn < t1:
            intervals.append(('momentum', t0, t_turn))
            intervals.append(('collapse', t_turn, t1))
        else:
            intervals.append((phase_name, t0, t1))

    # If turnaround happens after last recorded momentum start, and isCollapse
    # is true, mark remainder as collapse
    if is_collapse and t_turn is not None:
        # Check if collapse phase already added
        has_collapse = any(p == 'collapse' for p, _, _ in intervals)
        if not has_collapse:
            # Turnaround might coincide with end of data
            intervals.append(('collapse', t_turn, t[-1]))

    # --- Event times ---
    t_trans = None
    for p in phases:
        if str(p) == 'transition':
            idx = np.where(phases == p)[0]
            if len(idx) > 0:
                t_trans = t[idx[0]]
            break
    # More robust: find first 'transition' phase
    for ii, p in enumerate(phases):
        if str(p) == 'transition':
            t_trans = t[ii]
            break

    t_Rcloud = None
    if np.isfinite(rCloud) and rCloud > 0:
        crossing = np.where(R2 >= rCloud)[0]
        if len(crossing) > 0:
            t_Rcloud = t[crossing[0]]

    return {
        'intervals': intervals,
        't_trans': t_trans,
        't_turn': t_turn,
        't_Rcloud': t_Rcloud,
        't_end': t[-1],
        'outcome': outcome,
    }


def plot_phase_timeline(simulations: dict, output_dir: Path, fmt: str = 'pdf',
                        show: bool = False) -> None:
    """
    Annotated Gantt-style timeline of phase durations for density profile comparison.

    Minimalistic black-and-white style sized for a single A&A column (~88 mm).
    Thin bars with duration labels above; legend at top.
    """
    logger.info("Figure 5: Phase Timeline (annotated Gantt)")

    # Phase styles: (facecolor, hatch, edgecolor)
    PHASE_STYLE = {
        'energy':     dict(facecolor='white',   hatch=None,     edgecolor='black'),
        'transition': dict(facecolor='#cccccc', hatch=None,     edgecolor='black'),
        'momentum':   dict(facecolor='white',   hatch='/////',  edgecolor='black'),
        'collapse':   dict(facecolor='#666666', hatch=None,     edgecolor='black'),
    }

    # Order: alpha=0, -1, -2, then BE
    TRACK_ORDER = ['PL0', 'PL-1', 'PL-2', 'BE14']
    tags_present = [tag for tag in TRACK_ORDER if tag in simulations]
    n_tracks = len(tags_present)

    if n_tracks == 0:
        logger.warning("No simulations found for phase timeline. Skipping.")
        return

    # Extract phase info for all runs
    all_info = {}
    for tag in tags_present:
        all_info[tag] = _extract_phase_info(simulations[tag])

    # Print all extracted intervals and event times
    print("\n" + "=" * 80)
    print("Phase Timeline — Extracted Intervals and Event Times")
    print("=" * 80)
    for tag in tags_present:
        info = all_info[tag]
        s = get_style(tag)
        print(f"\n  {s['label']} ({tag}):")
        print(f"    Intervals:")
        for phase_name, t0, t1 in info['intervals']:
            print(f"      {phase_name:12s}  {t0:.4f} — {t1:.4f} Myr  (Δt = {t1-t0:.4f} Myr)")
        print(f"    t_trans  = {info['t_trans']}")
        print(f"    t_turn   = {info['t_turn']}")
        print(f"    t_Rcloud = {info['t_Rcloud']}")
        print(f"    t_end    = {info['t_end']:.4f} Myr")
        print(f"    Outcome  = {info['outcome']}")

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    bar_height = 0.1
    # Position bar bottoms at 0.1, 0.3, 0.5, 0.7 (centres at 0.15, 0.35, 0.55, 0.75)
    y_positions = np.array([0.1, 0.3, 0.5, 0.7])[:n_tracks]
    y_centres = y_positions + bar_height / 2

    t_max_global = max(info['t_end'] for info in all_info.values())

    for idx, tag in enumerate(tags_present):
        info = all_info[tag]
        yb = y_positions[idx]   # bar bottom
        yc = y_centres[idx]     # bar centre

        # Profile label left-aligned above the bar, with a small padding so
        # it doesn't sit flush against the y-axis spine.
        label_text = get_style(tag)['label'].replace('\n', ' ')
        t_left = info['intervals'][0][1] if info['intervals'] else 0.0
        t_pad = 0.01 * t_max_global
        ax.text(t_left + t_pad, yb - 0.015, label_text,
                ha='left', va='bottom', zorder=5,
                fontsize=plt.rcParams['font.size'] - 0.5)

        # Draw phase segments
        is_expanding = (info['outcome'] == 'expanding')
        n_intervals = len(info['intervals'])
        for seg_idx, (phase_name, t0, t1) in enumerate(info['intervals']):
            sty = PHASE_STYLE.get(phase_name, PHASE_STYLE['collapse'])
            is_last = (seg_idx == n_intervals - 1)

            if is_last and is_expanding:
                # Draw bar with no edge, then add edges manually
                from matplotlib.patches import Rectangle
                y_bot = yc - bar_height / 2
                rect = Rectangle((t0, y_bot), t1 - t0, bar_height,
                                 facecolor=sty['facecolor'], edgecolor='none',
                                 hatch=sty['hatch'], zorder=2)
                ax.add_patch(rect)
                # Solid edges: left, top, bottom
                ax.plot([t0, t0], [y_bot, y_bot + bar_height],
                        color='black', lw=0.5, zorder=3)
                ax.plot([t0, t1], [y_bot + bar_height, y_bot + bar_height],
                        color='black', lw=0.5, zorder=3)
                ax.plot([t0, t1], [y_bot, y_bot],
                        color='black', lw=0.5, zorder=3)
                # Dashed right edge
                ax.plot([t1, t1], [y_bot, y_bot + bar_height],
                        color='black', lw=0.8, ls='--', zorder=3)
            else:
                ax.barh(yc, t1 - t0, left=t0, height=bar_height, align='center',
                        facecolor=sty['facecolor'], edgecolor=sty['edgecolor'],
                        hatch=sty['hatch'], lw=0.5, zorder=2)


    # Y-axis: hide tick labels since profile names are drawn above each bar
    ax.set_yticks([])
    ax.invert_yaxis()  # top-to-bottom ordering
    ax.set_ylim(0.85, 0.0)

    # X-axis
    ax.set_xlabel(r'$t$ [Myr]')
    ax.set_xlim(0, t_max_global * 1.05)

    # Remove top/right spines for cleaner look; keep the left spine so the
    # y-axis line is visible (previously hidden).
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend at top (2x2 block instead of a single long row)
    legend_handles = [
        Patch(facecolor='white', edgecolor='black', lw=0.5,
              label='Energy-driven'),
        Patch(facecolor='#cccccc', edgecolor='black', lw=0.5,
              label='Transition'),
        Patch(facecolor='white', edgecolor='black', hatch='/////', lw=0.5,
              label='Momentum-driven'),
        Patch(facecolor='#666666', edgecolor='black', lw=0.5,
              label='Re-collapse'),
    ]
    ax.legend(handles=legend_handles, loc='lower center',
              bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False, columnspacing=1.0, handletextpad=0.4,
              handlelength=1.5)

    fig.tight_layout()

    savefig(fig, 'densityProfile_phaseTimeline', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)

    # --- Print phase duration summary table ---
    print("\n" + "=" * 100)
    print("Phase Duration Summary")
    print("=" * 100)
    header = f"{'Profile':<25s} | {'Energy [Myr]':>13s} | {'Trans [Myr]':>12s} | {'Momentum [Myr]':>15s} | {'Collapse [Myr]':>15s} | {'Total [Myr]':>12s} | {'Outcome':<12s}"
    print(header)
    print("-" * len(header))

    for tag in tags_present:
        info = all_info[tag]
        s = get_style(tag)
        label = s['label'].replace('$', '').replace('\\propto', '~').replace('\\rho', 'rho')

        # Sum durations by phase type
        durations = {'energy': 0.0, 'transition': 0.0, 'momentum': 0.0, 'collapse': 0.0}
        for phase_name, t0, t1 in info['intervals']:
            if phase_name in durations:
                durations[phase_name] += t1 - t0

        total = sum(durations.values())

        def fmt_dur(val):
            return f'{val:.3f}' if val > 0 else '---'

        print(f"{label:<25s} | {fmt_dur(durations['energy']):>13s} | {fmt_dur(durations['transition']):>12s} | {fmt_dur(durations['momentum']):>15s} | {fmt_dur(durations['collapse']):>15s} | {total:>12.3f} | {info['outcome']:<12s}")

    print("=" * 100 + "\n")


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
# Figure 6: Escape Fraction
# =============================================================================

def plot_escape_fraction(simulations: dict, output_dir: Path, fmt: str = 'pdf',
                         show: bool = False) -> None:
    """Plot ionising photon escape fraction f_esc(t) for all profiles.

    Uses the same masking/visualisation treatment as ``paper_escapeFraction``:
      - Suppress the seed-bubble transient before the shell first becomes
        optically thick (f_esc first reaches zero), showing that stretch as a
        faint dashed line at f_esc = 0.
      - Smooth the post-transient curve with the same moving-average window.
    """
    from src._plots.paper_escapeFraction import (
        load_escape_fraction, SMOOTH_WINDOW,
    )
    from src._plots.plot_base import smooth_1d

    logger.info("Figure 6: Escape Fraction")

    fig, ax = plt.subplots(figsize=(5, 4))

    for tag in PROFILE_ORDER:
        if tag not in simulations:
            continue
        output = simulations[tag]
        s = get_style(tag)

        data_path = getattr(output, 'filepath', None)
        try:
            if data_path is not None:
                t, fesc, _isCollapse, t_transient = load_escape_fraction(data_path)
            else:
                raise AttributeError("no filepath on TrinityOutput")
        except Exception:
            # Fallback: compute f_esc inline with the same transient mask.
            t_full = output.get('t_now')
            fAbs = safe_get(output, 'shell_fAbsorbedIon')
            fAbs = np.nan_to_num(fAbs, nan=0.0)
            fesc_full = 1.0 - fAbs
            t_transient = np.array([])
            idx_zero = np.nonzero(fesc_full <= 0.0)[0]
            if len(idx_zero) > 0:
                i0 = idx_zero[0]
                t_transient = t_full[:i0 + 1]
                t, fesc = t_full[i0:], fesc_full[i0:]
            else:
                t, fesc = t_full, fesc_full

        fesc_plot = smooth_1d(fesc, SMOOTH_WINDOW)
        fesc_plot = np.clip(fesc_plot, 0.0, 1.0)

        ax.plot(t, fesc_plot, color=s['color'], ls=s['ls'], lw=1.5)
        if len(t_transient) > 0:
            ax.plot(t_transient, np.zeros_like(t_transient),
                    color=s['color'], ls='--', lw=1.0, alpha=0.5)

    ax.set_xlabel(r'$t$ [Myr]')
    ax.set_ylabel(r'$f_{\rm esc,\,ion}$')
    ax.set_ylim(0, 1)

    add_legend(ax, [tag for tag in PROFILE_ORDER if tag in simulations], loc='best')

    savefig(fig, 'densityProfile_escapeFraction', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Figure 7: Feedback Fraction Grid (2x2)
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

    logger.info("Figure 7: Feedback Fraction Grid")

    sim_paths = _get_profile_data_paths(sweep_dir)
    tags_present = [tag for tag in PROFILE_ORDER if tag in sim_paths]

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
            t, R2, phase, base_forces, overlay_forces, rcloud, isCollapse, pressures = \
                _fb.load_run(sim_paths[tag])
            _fb.plot_run_on_ax(
                ax, t, R2, phase, base_forces, overlay_forces, rcloud,
                isCollapse, pressures=pressures, alpha=0.75,
                smooth_window=_fb.SMOOTH_WINDOW,
                phase_change=SHOW_PHASE, show_rcloud=SHOW_RCLOUD,
                show_collapse=SHOW_COLLAPSE, use_log_x=_fb.USE_LOG_X,
            )
        except Exception as e:
            logger.error(f"Feedback grid {tag}: {e}")
            ax.text(0.5, 0.5, "error", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue

        ax.set_title(s['label'].replace('\n', ' '))
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
        handles.append(Patch(facecolor='none', edgecolor=_fb.C_WIND,
                             hatch='\\\\\\\\', label='Wind'))
        handles.append(Patch(facecolor='none', edgecolor=_fb.C_SN,
                             hatch='\\\\\\\\', label='SN'))
    handles.extend(get_marker_legend_handles(
        include_phase=SHOW_PHASE, include_rcloud=SHOW_RCLOUD,
        include_collapse=SHOW_COLLAPSE))

    fig.legend(handles=handles, loc='upper center', ncol=4,
               frameon=True, facecolor='white', framealpha=0.9,
               edgecolor='0.2', bbox_to_anchor=(0.5, 1.05))
    fig.subplots_adjust(top=0.88)

    savefig(fig, 'densityProfile_feedback', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Figure 8: Momentum Grid (2x2)
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

    logger.info("Figure 8: Momentum Grid")

    sim_paths = _get_profile_data_paths(sweep_dir)
    tags_present = [tag for tag in PROFILE_ORDER if tag in sim_paths]

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
                smooth_window=_mom.SMOOTH_WINDOW, phase_change=SHOW_PHASE,
                show_rcloud=SHOW_RCLOUD, show_collapse=SHOW_COLLAPSE,
            )
        except Exception as e:
            logger.error(f"Momentum grid {tag}: {e}")
            ax.text(0.5, 0.5, "error", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue

        ax.set_title(s['label'].replace('\n', ' '))
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
    handles.append(Line2D([0], [0], color='darkgrey', lw=2.4, label='Net'))
    handles.extend(get_marker_legend_handles(
        include_phase=SHOW_PHASE, include_rcloud=SHOW_RCLOUD,
        include_collapse=SHOW_COLLAPSE))

    fig.legend(handles=handles, loc='upper center', ncol=4,
               frameon=True, facecolor='white', framealpha=0.9,
               edgecolor='0.2', bbox_to_anchor=(0.5, 1.05))
    fig.subplots_adjust(top=0.88)

    savefig(fig, 'densityProfile_momentum', output_dir, fmt)
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

    # Marker options (off by default for clean paper figures)
    marker_group = parser.add_argument_group(
        "markers", "Diagnostic markers (all off by default for clean figures)")
    marker_group.add_argument(
        '--show-phase', action='store_true', default=False,
        help='Show phase-transition markers (T / M vertical lines)')
    marker_group.add_argument(
        '--show-rcloud', action='store_true', default=False,
        help='Show R2 > R_cloud breakout marker')
    marker_group.add_argument(
        '--show-rcloud-horizontal', action='store_true', default=False,
        help='Show horizontal R_cloud line on radius panels')
    marker_group.add_argument(
        '--show-collapse', action='store_true', default=False,
        help='Show collapse onset marker')
    marker_group.add_argument(
        '--show-all-markers', action='store_true', default=False,
        help='Enable all diagnostic markers at once')

    args = parser.parse_args()

    # Wire CLI marker flags to module-level globals
    global SHOW_PHASE, SHOW_RCLOUD, SHOW_RCLOUD_H, SHOW_COLLAPSE
    _all = args.show_all_markers
    SHOW_PHASE = _all or args.show_phase
    SHOW_RCLOUD = _all or args.show_rcloud
    SHOW_RCLOUD_H = _all or args.show_rcloud_horizontal
    SHOW_COLLAPSE = _all or args.show_collapse

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

    # Figure 2 takes sweep_dir as well (for density-profile ingredients)
    try:
        plot_shell_evolution(simulations, output_dir, args.fmt, args.show,
                             sweep_dir=args.folder)
    except Exception as e:
        logger.error(f"Figure 2 (Shell Evolution) failed: {e}")

    # Figure 2p: paper-ready 2-panel version (ingredients + R_b(t))
    try:
        plot_shell_evolution_paper(simulations, output_dir, args.fmt, args.show,
                                   sweep_dir=args.folder)
    except Exception as e:
        logger.error(f"Figure 2p (Shell Evolution, paper) failed: {e}")

    # Generate all simulation-based figures
    plot_functions = [
        ("Figure 3: Pressure Budget",       plot_pressure_budget),
        ("Figure 4: Force Budget",          plot_force_budget),
        ("Figure 5: Phase Timing",          plot_phase_timing),
        ("Figure 6: Escape Fraction",       plot_escape_fraction),
    ]

    for name, func in plot_functions:
        try:
            func(simulations, output_dir, args.fmt, args.show)
        except Exception as e:
            logger.error(f"{name} failed: {e}")

    # Grid figures (reuse plot functions from paper_feedback/momentum/thermal)
    grid_functions = [
        ("Figure 7: Feedback Grid",   plot_feedback_grid),
        ("Figure 8: Momentum Grid",   plot_momentum_grid),
    ]

    for name, func in grid_functions:
        try:
            func(args.folder, output_dir, args.fmt, args.show)
        except Exception as e:
            logger.error(f"{name} failed: {e}")

    logger.info(f"All figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
