#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Density Profile Comparison Figures for TRINITY (paper version).

Compares four density profiles (same cloud mass, SFE, core density, but
varying density structure) from a density_profile_sweep run:
  - Power-law rho ~ r^0  (uniform)
  - Power-law rho ~ r^-1
  - Power-law rho ~ r^-2
  - Critical Bonnor-Ebert sphere

Produces two paper-ready figures:
  - densityProfile_paper          : top  rho(r) + M_enc(r) ingredients,
                                    bottom R_b(t) shell radius.
  - densityProfile_phaseTimeline  : Gantt-style phase timeline.

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
from src._plots.plot_markers import add_plot_markers

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
# Constants
# =============================================================================

# Big-font style for the Gantt phase-timeline figure only.  The paper
# figure (densityProfile_paper) inherits trinity.mplstyle defaults so it
# matches paper_teaser at column width.  The Gantt is a standalone full-
# page figure, so it keeps its own larger-font preset via rc_context().
_GANTT_RCPARAMS = {
    'font.size':        20,
    'axes.labelsize':   20,
    'axes.titlesize':   20,
    'xtick.labelsize':  18,
    'ytick.labelsize':  18,
    'legend.fontsize':  16,
}

# Colourblind-safe palette (Wong 2011) with solid lines for clarity
PROFILE_STYLES = {
    'PL0':  {'color': '#0072B2', 'ls': '-',  'label': r'$\rho \propto r^{0}$'},
    'PL-1': {'color': '#E69F00', 'ls': '-',  'label': r'$\rho \propto r^{-1}$'},
    'PL-2': {'color': '#CC79A7', 'ls': '-',  'label': r'$\rho \propto r^{-2}$'},
    'BE14': {'color': '#009E73', 'ls': '-',  'label': r'$\rho \propto \exp\{-\psi(\xi_{\rm cl})\}$'},
}

# Ordered list for consistent iteration
PROFILE_ORDER = ['PL0', 'PL-1', 'PL-2', 'BE14']


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
    # Match _PL{int} or _BE{int} either at end of folder name or before
    # a trailing _yesPHII / _noPHII (or any other) suffix.
    match = re.search(r'_(PL-?\d+|BE\d+)(?:_|$)', folder_name)
    if match:
        return match.group(1)
    return None


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
    # Mass-conversion mean molecular weight (mu_convert = 1.4 m_H,
    # independent of ionization state). This is what the rest of the
    # codebase uses for n -> rho — mu_ion (~0.61) counts ions+electrons
    # and is the wrong factor for a neutral / molecular cloud.
    mu_convert=1.4 * CGS.m_H * CONV.g2Msun,  # Msun
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
        mCloud, nCore, rCore, nISM, mu_convert   – all in internal units
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
        'mu_convert':    _try_float(raw.get('mu_convert'),    _DEFAULTS['mu_convert']),
        'dens_profile':  raw.get('dens_profile',              _DEFAULTS['dens_profile']),
        'densPL_alpha':  _try_float(raw.get('densPL_alpha'),  _DEFAULTS['densPL_alpha']),
        'densBE_Omega':  _try_float(raw.get('densBE_Omega'),  _DEFAULTS['densBE_Omega']),
    }


# Density conversion: internal [1/pc³] -> CGS [cm⁻³]
NDENS_AU2CGS = INV_CONV.ndens_au2cgs


# =============================================================================
# In-memory bundle assembly (folder → dict consumed by both drawers)
# =============================================================================
# Bundle structure (one density-profile sweep per bundle):
#   bundle['tags']            : list of tag strings, in PROFILE_ORDER
#   bundle[<tag>]             : per-tag dict with
#       t, R2, v2             : float arrays — timeseries used by both figs
#       phase                 : U16 array  — phase tag per snapshot
#       isCollapse            : bool array — collapse flag per snapshot
#       rCloud                : float scalar — cloud radius for this run [pc]
#       r_arr, n_cgs, M_arr   : profile ingredients for the top panel of
#                               figure 1 (precomputed so the bundle is
#                               decoupled from future cloud-shape library
#                               changes)
#       mu_g                  : float scalar — mu_convert in grams
def _build_bundle_from_folder(sweep_dir: str) -> dict:
    """Load every tag in *sweep_dir* into a single in-memory bundle.

    Reads each tag's TrinityOutput, extracts the timeseries fields both
    figures consume, then computes the ρ/M_enc profile ingredients via
    ``_compute_rho_M_profile`` so the bundle is fully self-contained.
    """
    sweep_path = Path(sweep_dir)
    sim_files = filter_sim_files_by_phii(
        find_all_simulations(sweep_path), "yes",
    )
    if not sim_files:
        raise FileNotFoundError(f"No density-profile runs found in {sweep_dir}")

    # Pair the data file with its parent folder per tag — ``load_output``
    # needs the dictionary.jsonl path, while ``_compute_rho_M_profile``
    # (via ``get_cloud_params``) reads ``*_summary.txt`` from the folder.
    data_paths = {}
    sim_folders = {}
    for data_path in sim_files:
        tag = identify_profile_tag(data_path.parent.name)
        if tag and tag in PROFILE_STYLES:
            data_paths[tag] = data_path
            sim_folders[tag] = data_path.parent

    bundle = {'tags': []}
    for tag in PROFILE_ORDER:
        if tag not in data_paths:
            continue
        try:
            output = load_output(data_paths[tag])
        except Exception as e:
            logger.warning(f"Skipping {tag}: load failed: {e}")
            continue

        phase_raw = output.get('current_phase', as_array=False)
        phase = (np.asarray([str(p) for p in phase_raw], dtype='U16')
                 if phase_raw is not None
                 else np.array([], dtype='U16'))

        isCollapse_raw = output.get('isCollapse', as_array=False)
        if isCollapse_raw is None:
            isCollapse = np.zeros(len(output), dtype=bool)
        else:
            isCollapse = np.array([bool(c) for c in isCollapse_raw], dtype=bool)

        rCloud_arr = safe_get(output, 'rCloud')
        rCloud_scalar = (float(rCloud_arr[-1])
                         if rCloud_arr.size > 0 and rCloud_arr[-1] > 0
                         else float('nan'))

        try:
            r_arr, n_cgs, M_arr, mu_g = _compute_rho_M_profile(tag, sim_folders)
        except Exception as e:
            logger.warning(f"Could not compute profile for {tag}: {e}")
            r_arr  = np.array([], dtype=float)
            n_cgs  = np.array([], dtype=float)
            M_arr  = np.array([], dtype=float)
            mu_g   = 0.0

        bundle[tag] = dict(
            t          = np.asarray(output.get('t_now'), dtype=float),
            R2         = safe_get(output, 'R2').astype(float),
            v2         = safe_get(output, 'v2').astype(float),
            phase      = phase,
            isCollapse = isCollapse,
            rCloud     = rCloud_scalar,
            r_arr      = np.asarray(r_arr, dtype=float),
            n_cgs      = np.asarray(n_cgs, dtype=float),
            M_arr      = np.asarray(M_arr, dtype=float),
            mu_g       = float(mu_g),
        )
        bundle['tags'].append(tag)

    if not bundle['tags']:
        raise ValueError(f"No usable density-profile runs in {sweep_dir}")
    return bundle


# =============================================================================
# .npz bundle: write, read, plot
# =============================================================================
_BUNDLE_TS_KEYS      = ('t', 'R2', 'v2', 'phase', 'isCollapse')
_BUNDLE_SCALAR_KEYS  = ('rCloud', 'mu_g')
_BUNDLE_PROFILE_KEYS = ('r_arr', 'n_cgs', 'M_arr')


def export_densityProfile_npz(sweep_dir, out_path) -> Path:
    """Reduce a density_profile_sweep folder to a single self-contained ``.npz``.

    Stores everything both figures consume: per-tag timeseries
    (t/R2/v2/phase/isCollapse/rCloud), the precomputed ρ/M_enc profile
    ingredients (r_arr, n_cgs, M_arr, mu_g), and the list of tags
    present. Original run folders + cloud-property summaries can then
    be discarded for paper reproducibility.
    """
    bundle = _build_bundle_from_folder(str(sweep_dir))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {'tags': np.array(bundle['tags'], dtype='U8')}
    for tag in bundle['tags']:
        e = bundle[tag]
        payload[f'{tag}_t']          = e['t']
        payload[f'{tag}_R2']         = e['R2']
        payload[f'{tag}_v2']         = e['v2']
        payload[f'{tag}_phase']      = e['phase']
        payload[f'{tag}_isCollapse'] = e['isCollapse']
        payload[f'{tag}_rCloud']     = np.float64(e['rCloud'])
        payload[f'{tag}_r_arr']      = e['r_arr']
        payload[f'{tag}_n_cgs']      = e['n_cgs']
        payload[f'{tag}_M_arr']      = e['M_arr']
        payload[f'{tag}_mu_g']       = np.float64(e['mu_g'])

    np.savez(out_path, **payload)
    logger.info(f"Exported: {out_path}")
    print(f"Exported: {out_path}")
    return out_path


def _build_bundle_from_npz(npz_path) -> dict:
    """Reconstruct the in-memory bundle from a published ``.npz``."""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        tags = [str(t) for t in z['tags']]
        bundle = {'tags': tags}
        for tag in tags:
            bundle[tag] = dict(
                t          = z[f'{tag}_t'].astype(float),
                R2         = z[f'{tag}_R2'].astype(float),
                v2         = z[f'{tag}_v2'].astype(float),
                phase      = np.asarray(z[f'{tag}_phase']),
                isCollapse = z[f'{tag}_isCollapse'].astype(bool),
                rCloud     = float(z[f'{tag}_rCloud']),
                r_arr      = z[f'{tag}_r_arr'].astype(float),
                n_cgs      = z[f'{tag}_n_cgs'].astype(float),
                M_arr      = z[f'{tag}_M_arr'].astype(float),
                mu_g       = float(z[f'{tag}_mu_g']),
            )
    return bundle


def plot_densityProfile_from_npz(npz_path, output_dir: Path,
                                 fmt: str = 'pdf', show: bool = False) -> None:
    """Reproduce both figures straight from a published bundle."""
    bundle = _build_bundle_from_npz(npz_path)
    plot_shell_evolution_paper(bundle, output_dir, fmt, show)
    plot_phase_timeline(bundle, output_dir, fmt, show)


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
    """Return (r_arr [pc], n_cgs [cm^-3], M_arr [Msun], mu_g [g])
    for profile *tag*.  ``mu_g`` is the mass-conversion mean molecular
    weight (mu_convert ≈ 1.4 m_H) expressed in grams, so
    ``rho_cgs = n_cgs * mu_g`` gives the cloud mass density in g/cm³.
    Using mu_convert here (rather than mu_ion or mu_atom) keeps the
    n↔ρ mapping ionization-state-independent, matching the rest of
    TRINITY (mass_profile.py, powerLawSphere, bonnorEbertSphere, …).
    """
    ptype_default, alpha_default, omega_default = _PROFILE_DEFAULTS[tag]

    if tag in sim_folders:
        cp = get_cloud_params(sim_folders[tag])
    else:
        cp = dict(_DEFAULTS)

    mCloud  = cp['mCloud']
    nCore   = cp['nCore']          # 1/pc³
    rCore   = cp['rCore']          # pc
    nISM    = cp['nISM']           # 1/pc³
    mu_au   = cp['mu_convert']     # Msun (state-independent n -> rho factor)
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

    mu_g = mu_au * INV_CONV.Msun2g
    return r_arr, n_arr * NDENS_AU2CGS, M_arr, mu_g


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
# densityProfile_paper: helpers (ticks, ingredients panel, R_b panel)
# =============================================================================

# Alpha for the dashed M_enc line on the top panel.
_MENC_ALPHA = 0.45


def _setup_time_panel_ticks(ax):
    """Apply consistent inward major/minor ticks on all four sides."""
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, right=True)
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', length=3)


def _draw_ingredients_panel(ax_rho, ax_M, bundle: dict,
                            tags_present: list) -> None:
    """Draw the density profile (solid) + M_enc (dashed, twinx) top panel.

    Configures ticks, scales, axis labels and the in-panel solid/dashed
    legend. y-axes are plotted as log10 values on a linear scale so that
    minor ticks are legible across many decades; x stays on a log scale.

    Profile ingredients (r_arr, n_cgs, M_arr, mu_g) come from the bundle;
    they were computed at load time so the figure stays reproducible if
    the underlying profile-shape libraries (Bonnor-Ebert, power-law)
    ever evolve.
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

    profiles = {}
    for tag in tags_present:
        entry = bundle.get(tag)
        if not entry or entry['r_arr'].size == 0:
            continue
        profiles[tag] = (entry['r_arr'], entry['n_cgs'],
                         entry['M_arr'], entry['mu_g'])
    if not profiles:
        return
    r_max = max(r_arr[-1] for r_arr, _, _, _ in profiles.values())

    for tag in tags_present:
        if tag not in profiles:
            continue
        s = get_style(tag)
        r_arr, n_cgs, M_arr, mu_g = profiles[tag]
        r_arr, n_cgs, M_arr = _extend_outer_plateau(r_arr, n_cgs, M_arr, r_max)
        # rho = n * mu_convert (in grams) — state-independent mass
        # conversion, matching the rest of TRINITY.
        rho_cgs = n_cgs * mu_g
        with np.errstate(divide='ignore', invalid='ignore'):
            log_rho = np.log10(rho_cgs)
            log_M = np.log10(M_arr)
        ax_rho.plot(r_arr, log_rho, color=s['color'], ls='-', lw=1.5)
        ax_M.plot(r_arr, log_M, color=s['color'], ls='--', lw=1.2,
                  alpha=_MENC_ALPHA)

    ax_rho.set_xscale('log')
    ax_M.set_xscale('log')
    # Display range starts at 1e-2 pc, not the inner 1e-3 used to compute
    # the profiles — there is little structure in the inner-most decade.
    ax_rho.set_xlim(left=1e-2, right=r_max)

    ax_rho.set_xlabel(r'$r$ [pc]')
    ax_rho.set_ylabel(r'$\log_{10}\!\left(\rho_{\rm cloud}(r)\right)$ [g cm$^{-3}$]')
    # Twiny label: same reading orientation (bottom-to-top) as the
    # primary y-axis label.
    ax_M.set_ylabel(
        r'$\log_{10}\!\left(M_{\rm enc}(<r)\right)$ [M$_\odot$]',
        rotation=90, labelpad=15, va='bottom',
    )

    style_handles = [
        Line2D([0], [0], color='black', ls='-',  lw=1.5,
               label=r'$\rho_{\rm cloud}(r)$'),
        Line2D([0], [0], color='black', ls='--', lw=1.2, alpha=_MENC_ALPHA,
               label=r'$M_{\rm enc}(<r)$'),
    ]
    ax_rho.legend(handles=style_handles, loc='upper left', frameon=False,
                  handlelength=2.0, handletextpad=0.5)


def _draw_Rb_panel(ax, bundle: dict, tags_present: list) -> None:
    """Draw the R_b(t) panel (outer bubble radius) with phase markers."""
    _setup_time_panel_ticks(ax)
    for tag in tags_present:
        entry = bundle[tag]
        s = get_style(tag)

        t = entry['t']
        R2 = entry['R2']
        phase = entry['phase']
        isCollapse = entry['isCollapse']
        rCloud_val = (float(entry['rCloud'])
                      if np.isfinite(entry['rCloud']) and entry['rCloud'] > 0
                      else None)

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


# =============================================================================
# densityProfile_paper: ingredients (rho, M_enc) + R_b(t)
# =============================================================================

def plot_shell_evolution_paper(bundle: dict, output_dir: Path,
                               fmt: str = 'pdf', show: bool = False) -> None:
    """Paper-ready 2-panel figure: density+M_enc ingredients (top) and
    R_b(t) shell radius (bottom).

    The profile-colour legend lives in the upper-left of the R_b
    panel as a single column; the rho/M_enc style legend lives inside
    the top panel; a generous ``hspace`` keeps the ingredient-panel
    xlabel clear of the R_b panel below.
    """
    logger.info("densityProfile_paper: ingredients + R_b(t)")

    tags_present = [tag for tag in PROFILE_ORDER if tag in bundle['tags']]

    # Column-width canvas matching paper_teaser (4.0").  Two stacked
    # panels at ~2.75" each, slightly taller per panel than teaser's
    # 3-panel 6.5" layout because we have one fewer panel to fit.
    fig = plt.figure(figsize=(4.0, 5.5))

    # Wide hspace because the top panel's x-axis (r [pc], log) is
    # different from the bottom's (t [Myr]) — they don't share x, so
    # the inner xlabel needs room.  Side margins account for the twinx
    # M_enc label on the right of the top panel.
    gs = fig.add_gridspec(
        2, 1,
        hspace=0.30,
        left=0.16, right=0.84,
        top=0.95, bottom=0.10,
    )
    ax_rho = fig.add_subplot(gs[0, 0])
    ax_M   = ax_rho.twinx()
    ax_R   = fig.add_subplot(gs[1, 0])

    _draw_ingredients_panel(ax_rho, ax_M, bundle, tags_present)
    _draw_Rb_panel(ax_R, bundle, tags_present)

    ax_R.set_xlabel(r'$t$ [Myr]')

    # Profile-colour legend in the upper-left of the R_b panel,
    # single column, no frame.
    profile_handles = [
        Line2D([0], [0], color=get_style(tag)['color'], ls='-', lw=1.8,
               label=get_style(tag)['label'].replace('\n', ' '))
        for tag in tags_present
    ]
    ax_R.legend(handles=profile_handles,
                loc='upper left', bbox_to_anchor=(0.01, 0.99),
                ncol=1, frameon=True,
                fontsize=10, handletextpad=0.5)

    savefig(fig, 'densityProfile_paper', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# densityProfile_phaseTimeline: phase intervals helper + Gantt timeline
# =============================================================================

def _extract_phase_info(entry: dict) -> dict:
    """
    Extract phase intervals and key event times from a bundle entry.

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
    t = entry['t']
    phases = entry['phase']
    v2 = entry['v2']
    R2 = entry['R2']
    isCollapse = entry['isCollapse']
    rCloud = float(entry['rCloud'])

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


def plot_phase_timeline(bundle: dict, output_dir: Path, fmt: str = 'pdf',
                        show: bool = False) -> None:
    """
    Annotated Gantt-style timeline of phase durations for density profile comparison.

    Minimalistic black-and-white style sized for a single A&A column (~88 mm).
    Thin bars with duration labels above; legend at top.
    """
    logger.info("densityProfile_phaseTimeline: annotated Gantt timeline")

    # Phase styles: (facecolor, hatch, edgecolor)
    PHASE_STYLE = {
        'energy':     dict(facecolor='white',   hatch=None,     edgecolor='black'),
        'transition': dict(facecolor='#cccccc', hatch=None,     edgecolor='black'),
        'momentum':   dict(facecolor='white',   hatch='/////',  edgecolor='black'),
        'collapse':   dict(facecolor='#666666', hatch=None,     edgecolor='black'),
    }

    # Order: alpha=0, -1, -2, then BE
    TRACK_ORDER = ['PL0', 'PL-1', 'PL-2', 'BE14']
    tags_present = [tag for tag in TRACK_ORDER if tag in bundle['tags']]
    n_tracks = len(tags_present)

    if n_tracks == 0:
        logger.warning("No simulations found for phase timeline. Skipping.")
        return

    # Extract phase info for all runs
    all_info = {}
    for tag in tags_present:
        all_info[tag] = _extract_phase_info(bundle[tag])

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
    # Gantt is a standalone full-page figure (not column-width like the
    # paper figure), so it gets its own larger-font preset via rc_context
    # — the rest of this script inherits trinity.mplstyle defaults.
    with plt.rc_context(_GANTT_RCPARAMS):
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
                  label='energy-driven'),
            Patch(facecolor='#cccccc', edgecolor='black', lw=0.5,
                  label='transition'),
            Patch(facecolor='white', edgecolor='black', hatch='/////', lw=0.5,
                  label='momentum-driven'),
            Patch(facecolor='#666666', edgecolor='black', lw=0.5,
                  label='re-collapse'),
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

  # Collapse a sweep folder to one self-contained .npz bundle (recommended
  # for paper reproducibility):
  python paper_densityProfile.py -F outputs/density_profile_sweep \\
      --export paper/data/densityProfile.npz

  # Reproduce both figures from a published bundle:
  python paper_densityProfile.py --from-npz paper/data/densityProfile.npz
        """
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '--folder', '-F',
        help='Path to density profile sweep output directory'
    )
    source.add_argument(
        '--from-npz', dest='from_npz',
        help='Reproduce both figures from a .npz bundle written by --export'
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help='Directory to save figures (default: fig/<folder_name>/)'
    )
    parser.add_argument(
        '--fmt', default='pdf',
        help='Figure format (default: pdf)'
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Show figures interactively instead of just saving'
    )
    parser.add_argument(
        '--export', default=None,
        help='Export the sweep folder to this .npz bundle and exit '
             '(no plot). Bundle contains per-tag timeseries plus the '
             'precomputed ρ / M_enc profile ingredients, so the original '
             'sweep folder can be discarded. Recommended location: paper/data/.'
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

    # --export short-circuits before any plotting.
    if args.export:
        if not args.folder:
            parser.error("--export requires --folder (cannot re-export a bundle)")
        try:
            export_densityProfile_npz(args.folder, args.export)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Export failed: {e}")
        return

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if args.folder:
            source_name = Path(args.folder).name
        else:
            source_name = Path(args.from_npz).stem
        output_dir = FIG_DIR / source_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the in-memory bundle (one shape, two sources).
    try:
        if args.from_npz:
            bundle = _build_bundle_from_npz(args.from_npz)
        else:
            bundle = _build_bundle_from_folder(args.folder)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Could not load bundle: {e}")
        return

    if not bundle['tags']:
        logger.error("No valid simulations found. Exiting.")
        return

    # densityProfile_paper: ingredients (rho, M_enc) + R_b(t)
    try:
        plot_shell_evolution_paper(bundle, output_dir, args.fmt, args.show)
    except Exception as e:
        logger.error(f"densityProfile_paper failed: {e}")

    # densityProfile_phaseTimeline: Gantt-style phase timeline
    try:
        plot_phase_timeline(bundle, output_dir, args.fmt, args.show)
    except Exception as e:
        logger.error(f"densityProfile_phaseTimeline failed: {e}")

    logger.info(f"All figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
