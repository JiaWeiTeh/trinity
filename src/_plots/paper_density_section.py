#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Density Profile Comparison: Scalar Diagnostics, Summary Table, and Figures.

Loads four density-profile TRINITY runs (PL0, PL-1, PL-2, BE) from a sweep
directory, extracts time-series and scalar diagnostics, prints a human-
readable comparison table, generates a 3x2 multi-panel figure, and writes
a CSV of extracted numbers.

Usage:
  python paper_density_section.py -F /path/to/density_profile_sweep/
  python paper_density_section.py -F outputs/density_profile_sweep --fmt png --show

@author: Jia Wei Teh
"""

import sys
import os
import re
import csv
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import (
    load_output, find_all_simulations, TrinityOutput,
)
from src._functions.unit_conversions import CONV, INV_CONV, CGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Load matplotlib style
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'trinity.mplstyle'))

# Output figure directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Constants and unit conversions (from unit_conversions, not hardcoded)
# =============================================================================
V_AU2KMS = INV_CONV.v_au2kms       # pc/Myr -> km/s
E_AU2ERG = INV_CONV.E_au2cgs       # Msun pc^2/Myr^2 -> erg
F_AU2CGS = INV_CONV.F_au2cgs       # Msun pc/Myr^2 -> dyn
PB_AU2CGS = INV_CONV.Pb_au2cgs     # Msun/Myr^2/pc -> dyn/cm^2
K_B_CGS = CGS.k_B                  # erg/K

# Profile style definitions (consistent with paper_densityProfile.py)
PROFILE_STYLES = {
    'PL0':  {'color': '#0072B2', 'ls': '-',   'label': r'$\alpha=0$ (uniform)'},
    'PL-1': {'color': '#D55E00', 'ls': '--',  'label': r'$\alpha=-1$'},
    'PL-2': {'color': '#009E73', 'ls': '-.',  'label': r'$\alpha=-2$ (SIS)'},
    'BE14': {'color': '#CC79A7', 'ls': ':',   'label': r'BE $\Omega=14.1$'},
}
PROFILE_ORDER = ['PL0', 'PL-1', 'PL-2', 'BE14']

# Mapping for the table header (short labels)
SHORT_LABELS = {
    'PL0': 'alpha=0', 'PL-1': 'alpha=-1',
    'PL-2': 'alpha=-2', 'BE14': 'BE (xi=6.45)',
}


# =============================================================================
# Helpers (mirrored from paper_densityProfile.py)
# =============================================================================

def identify_profile_tag(folder_name: str):
    """Return profile tag (PL0, PL-1, PL-2, BE14) from folder name."""
    match = re.search(r'_(PL-?\d+|BE\d+)$', folder_name)
    return match.group(1) if match else None


def safe_get(output: TrinityOutput, key: str,
             default_val: float = 0.0) -> np.ndarray:
    """Get a field from TrinityOutput, returning default array if missing."""
    try:
        values = output.get(key, as_array=False)
    except Exception:
        return np.full(len(output), default_val)
    if values is None:
        return np.full(len(output), default_val)
    result = np.array(
        [default_val if v is None else float(v) for v in values],
        dtype=float,
    )
    return np.where(np.isfinite(result), result, default_val)


def load_sweep(sweep_dir: str) -> dict:
    """
    Discover and load all simulations in *sweep_dir*.

    Returns
    -------
    dict  tag -> TrinityOutput
    """
    sweep_path = Path(sweep_dir)
    sim_files = find_all_simulations(sweep_path)
    if not sim_files:
        raise FileNotFoundError(f"No simulations found in {sweep_path}")

    simulations = {}
    for data_path in sim_files:
        folder_name = data_path.parent.name
        tag = identify_profile_tag(folder_name)
        if tag is None or tag not in PROFILE_STYLES:
            logger.warning(f"Skipping unrecognised folder: {folder_name}")
            continue
        logger.info(f"Loading {tag}: {data_path}")
        simulations[tag] = load_output(data_path)

    logger.info(f"Loaded {len(simulations)} simulations: {list(simulations)}")
    return simulations


def get_style(tag: str) -> dict:
    return PROFILE_STYLES.get(tag, {'color': 'gray', 'ls': '-', 'label': tag})


def savefig(fig, name: str, output_dir: Path, fmt: str = 'pdf'):
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{name}.{fmt}"
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    logger.info(f"Saved: {filepath}")
    return filepath


def add_legend(ax, tags, **kwargs):
    handles = []
    for tag in tags:
        s = get_style(tag)
        handles.append(Line2D([0], [0], color=s['color'], ls=s['ls'],
                              lw=1.5, label=s['label']))
    ax.legend(handles=handles, **kwargs)


# =============================================================================
# Time-series extraction
# =============================================================================

def extract_timeseries(output: TrinityOutput) -> dict:
    """
    Return a dict of time-series arrays (all in physical units).
    """
    t = output.get('t_now')                            # Myr
    R = safe_get(output, 'R2')                         # pc
    v = safe_get(output, 'v2') * V_AU2KMS              # km/s
    mshell = safe_get(output, 'shell_mass')            # Msun
    p = mshell * v                                     # Msun km/s
    Eb = safe_get(output, 'Eb') * E_AU2ERG             # erg
    Lmech = safe_get(output, 'Lmech_total')            # code (Msun pc^2/Myr^3)
    rCloud = safe_get(output, 'rCloud')                # pc

    # Cumulative energy input  [erg]
    E_input_code = np.zeros_like(t)
    if np.any(Lmech > 0):
        E_input_code[1:] = np.cumsum(
            0.5 * (Lmech[1:] + Lmech[:-1]) * np.diff(t))
    E_input = E_input_code * E_AU2ERG

    # Phase string array
    try:
        phases = output.get('current_phase', as_array=False)
        if phases is None:
            phases = [''] * len(t)
    except Exception:
        phases = [''] * len(t)

    # Pressure terms  [dyn/cm^2]
    Pb = np.abs(safe_get(output, 'Pb')) * PB_AU2CGS
    P_drive = np.abs(safe_get(output, 'P_drive')) * PB_AU2CGS
    P_IF = np.abs(safe_get(output, 'P_IF')) * PB_AU2CGS

    # Radiation pressure = F_rad / (4 pi R^2) converted to cgs
    R_cm = R * INV_CONV.pc2cm
    F_rad_cgs = np.abs(safe_get(output, 'F_rad')) * F_AU2CGS
    with np.errstate(divide='ignore', invalid='ignore'):
        P_rad = np.where(R_cm > 0,
                         F_rad_cgs / (4.0 * np.pi * R_cm ** 2), 0.0)

    return {
        't': t, 'R': R, 'v': v, 'mshell': mshell, 'p': p,
        'Eb': Eb, 'E_input': E_input, 'rCloud': rCloud,
        'Pb': Pb, 'P_drive': P_drive, 'P_IF': P_IF, 'P_rad': P_rad,
        'phases': phases,
    }


# =============================================================================
# Scalar diagnostics
# =============================================================================

def _interp_at(t_arr, y_arr, t_target):
    """Linearly interpolate y at t_target.  NaN if outside range."""
    if t_target < t_arr[0] or t_target > t_arr[-1]:
        return np.nan
    return float(np.interp(t_target, t_arr, y_arr))


def extract_scalars(ts: dict) -> dict:
    """
    Compute scalar diagnostics from a time-series dict.

    Returns a flat dict of diagnostic values.
    """
    t, R, v, p = ts['t'], ts['R'], ts['v'], ts['p']
    Eb, E_input = ts['Eb'], ts['E_input']
    phases = ts['phases']

    sc = {}

    # Values at fixed times
    for t_ref in [1.0, 3.0]:
        label = f'{t_ref:.0f}'
        sc[f'R_{label}Myr'] = _interp_at(t, R, t_ref)
        sc[f'v_{label}Myr'] = _interp_at(t, v, t_ref)
        sc[f'p_{label}Myr'] = _interp_at(t, np.abs(p), t_ref)
        sc[f'mshell_{label}Myr'] = _interp_at(t, ts['mshell'], t_ref)

    # Energy retention at 1 Myr
    Eb_1 = _interp_at(t, Eb, 1.0)
    Einp_1 = _interp_at(t, E_input, 1.0)
    if np.isfinite(Eb_1) and np.isfinite(Einp_1) and Einp_1 > 0:
        sc['E_retain_1Myr'] = Eb_1 / Einp_1
    else:
        sc['E_retain_1Myr'] = np.nan

    # Maximum shell radius and velocity
    sc['R_max'] = float(np.max(R))
    sc['v_max'] = float(np.max(v))

    # Final momentum
    sc['p_final'] = float(np.abs(p[-1]))
    sc['t_end'] = float(t[-1])

    # Phase transition time (energy/implicit -> transition or momentum)
    t_transition = np.nan
    for i in range(1, len(phases)):
        prev = str(phases[i - 1])
        cur = str(phases[i])
        if prev in ('energy', 'implicit') and cur not in ('energy', 'implicit'):
            t_transition = float(t[i])
            break
    sc['t_transition'] = t_transition

    # Stall time: first time v <= 0 after expansion starts
    stall_mask = v <= 0
    if np.any(stall_mask):
        sc['t_stall'] = float(t[stall_mask][0])
    else:
        sc['t_stall'] = np.nan

    # Outcome: dispersal vs. re-collapse
    # Simple heuristic: if R at end > 0.9 * rCloud -> dispersed
    rCloud_end = ts['rCloud'][-1] if ts['rCloud'].size > 0 else np.inf
    if R[-1] >= 0.9 * rCloud_end:
        sc['outcome'] = 'dispersed'
    elif np.any(stall_mask):
        sc['outcome'] = 're-collapse'
    else:
        sc['outcome'] = 'expanding'

    return sc


# =============================================================================
# Summary table printing
# =============================================================================

def print_summary_table(all_scalars: dict, tags: list):
    """Print a formatted comparison table to stdout."""
    col_w = 14

    header_labels = [SHORT_LABELS.get(t, t) for t in tags]
    header = f"{'Quantity':<30s} | " + " | ".join(
        f"{h:^{col_w}s}" for h in header_labels)
    sep = '-' * 30 + '-+-' + '-+-'.join('-' * col_w for _ in tags)

    print()
    print("=" * len(sep))
    print("  TRINITY Density Profile Comparison")
    print("  Cloud: M_cl = 1e5 Msun, n0 = 1e4 cm^-3, SFE = 0.01")
    print("=" * len(sep))
    print(header)
    print(sep)

    rows = [
        ('R at 1 Myr [pc]',          'R_1Myr',        '{:.3f}'),
        ('v at 1 Myr [km/s]',        'v_1Myr',        '{:.2f}'),
        ('p at 1 Myr [Msun km/s]',   'p_1Myr',        '{:.1f}'),
        ('M_sh at 1 Myr [Msun]',     'mshell_1Myr',   '{:.0f}'),
        ('R at 3 Myr [pc]',          'R_3Myr',        '{:.3f}'),
        ('v at 3 Myr [km/s]',        'v_3Myr',        '{:.2f}'),
        ('p at 3 Myr [Msun km/s]',   'p_3Myr',        '{:.1f}'),
        ('R_max [pc]',               'R_max',         '{:.3f}'),
        ('v_max [km/s]',             'v_max',         '{:.2f}'),
        ('p_final [Msun km/s]',      'p_final',       '{:.1f}'),
        ('E_retain at 1 Myr [%]',    'E_retain_1Myr', '{:.1%}'),
        ('t_transition [Myr]',       't_transition',  '{:.3f}'),
        ('t_stall [Myr]',            't_stall',       '{:.3f}'),
        ('t_end [Myr]',              't_end',         '{:.3f}'),
        ('Outcome',                  'outcome',       '{}'),
    ]

    for label, key, fmt in rows:
        vals = []
        for tag in tags:
            v = all_scalars[tag].get(key, np.nan)
            if isinstance(v, str):
                vals.append(f"{v:^{col_w}s}")
            elif np.isfinite(v):
                vals.append(f"{fmt.format(v):^{col_w}s}")
            else:
                vals.append(f"{'N/A':^{col_w}s}")
        print(f"{label:<30s} | " + " | ".join(vals))

    print(sep)

    # ------------------------------------------------------------------
    # Ratios relative to PL0 (uniform)
    # ------------------------------------------------------------------
    baseline = 'PL0'
    if baseline not in all_scalars:
        return
    base = all_scalars[baseline]

    print()
    print("=== Ratios relative to uniform (alpha=0) ===")
    ratio_keys = [
        ('R(1 Myr)', 'R_1Myr'), ('v(1 Myr)', 'v_1Myr'),
        ('p(1 Myr)', 'p_1Myr'), ('R_max', 'R_max'),
        ('p_final', 'p_final'),
    ]
    for label, key in ratio_keys:
        base_v = base.get(key, np.nan)
        if not np.isfinite(base_v) or base_v == 0:
            continue
        parts = [f"{label:>12s} ratio:"]
        for tag in tags:
            if tag == baseline:
                continue
            v = all_scalars[tag].get(key, np.nan)
            if np.isfinite(v):
                parts.append(f"  {SHORT_LABELS[tag]}: {v / base_v:.2f}x")
            else:
                parts.append(f"  {SHORT_LABELS[tag]}: N/A")
        print("  ".join(parts))
    print()


# =============================================================================
# Multi-panel figure
# =============================================================================

def plot_comparison(all_ts: dict, tags: list, output_dir: Path,
                    fmt: str = 'pdf', show: bool = False):
    """
    3x2 panel figure:
      (a) R(t)   (b) v(t)
      (c) p(t)   (d) M_sh(t)
      (e) Eb(t)  (f) Pressure terms (PL0 only)
    """
    fig, axes = plt.subplots(3, 2, figsize=(8, 9))

    # Determine common time range
    t_max = min(ts['t'][-1] for ts in all_ts.values())

    # Panels (a)-(e): overlay all profiles
    panel_defs = [
        (axes[0, 0], 'R',      r'$R$ [pc]',
         False, '(a) Shell Radius'),
        (axes[0, 1], 'v',      r'$v$ [km\,s$^{-1}$]',
         True,  '(b) Shell Velocity'),
        (axes[1, 0], 'p',      r'$p$ [M$_\odot$\,km\,s$^{-1}$]',
         True,  '(c) Radial Momentum'),
        (axes[1, 1], 'mshell', r'$M_{\rm shell}$ [M$_\odot$]',
         False, '(d) Swept-up Mass'),
        (axes[2, 0], 'Eb',     r'$E_{\rm b}$ [erg]',
         True,  '(e) Bubble Energy'),
    ]

    for ax, key, ylabel, use_log, title in panel_defs:
        for tag in tags:
            ts = all_ts[tag]
            s = get_style(tag)
            t = ts['t']
            y = np.abs(ts[key]) if key == 'p' else ts[key]

            mask = t <= t_max
            if use_log:
                pos = y > 0
                combined = mask & pos
                if np.any(combined):
                    ax.semilogy(t[combined], y[combined],
                                color=s['color'], ls=s['ls'], lw=1.5)
            else:
                ax.plot(t[mask], y[mask],
                        color=s['color'], ls=s['ls'], lw=1.5)

        ax.set_xlabel(r'$t$ [Myr]')
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', fontsize=12)

    # Panel (f): Pressure terms for PL0 (or first available)
    ax_p = axes[2, 1]
    pressure_tag = tags[0]
    ts_p = all_ts[pressure_tag]
    t_p = ts_p['t']
    mask_p = t_p <= t_max

    pressure_fields = [
        ('Pb',      r'$P_{\rm b}$',     '#0072B2', '-'),
        ('P_drive', r'$P_{\rm drive}$', '#D55E00', '--'),
        ('P_IF',    r'$P_{\rm IF}$',    '#009E73', '-.'),
        ('P_rad',   r'$P_{\rm rad}$',   '#CC79A7', ':'),
    ]
    for key, label, color, ls in pressure_fields:
        y = ts_p[key]
        pos = (y > 0) & mask_p
        if np.any(pos):
            ax_p.semilogy(t_p[pos], y[pos], color=color, ls=ls, lw=1.2,
                          label=label)

    ax_p.set_xlabel(r'$t$ [Myr]')
    ax_p.set_ylabel(r'$P$ [dyn\,cm$^{-2}$]')
    ax_p.set_title(
        f'(f) Pressures ({SHORT_LABELS.get(pressure_tag, pressure_tag)})',
        loc='left', fontsize=12)
    ax_p.legend(fontsize=9, loc='best')

    # Shared legend for panels (a)-(e)
    add_legend(axes[0, 0], tags, loc='best', fontsize=9)

    fig.tight_layout()
    savefig(fig, 'paper_density_section', output_dir, fmt)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# CSV export
# =============================================================================

def write_csv(all_scalars: dict, tags: list, output_dir: Path):
    """Write scalar diagnostics to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / 'density_profile_comparison.csv'

    # Collect all keys from the first tag
    all_keys = list(next(iter(all_scalars.values())).keys())

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['quantity'] + [SHORT_LABELS.get(t, t) for t in tags])
        for key in all_keys:
            row = [key]
            for tag in tags:
                val = all_scalars[tag].get(key, '')
                if isinstance(val, float) and np.isfinite(val):
                    row.append(f'{val:.6g}')
                else:
                    row.append(str(val))
            writer.writerow(row)

    logger.info(f"Saved CSV: {filepath}")
    return filepath


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TRINITY Density Profile Comparison (Section Diagnostics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_density_section.py -F outputs/density_profile_sweep
  python paper_density_section.py -F outputs/density_profile_sweep --fmt png
  python paper_density_section.py -F outputs/density_profile_sweep --show
        """,
    )
    parser.add_argument(
        '--folder', '-F', required=True,
        help='Path to density profile sweep output directory',
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help='Directory to save figures and CSV (default: fig/<folder_name>/)',
    )
    parser.add_argument(
        '--fmt', default='pdf',
        help='Figure format (default: pdf)',
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Show figures interactively',
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load simulations
    # ------------------------------------------------------------------
    simulations = load_sweep(args.folder)
    if not simulations:
        logger.error("No valid simulations found. Exiting.")
        return

    tags = [t for t in PROFILE_ORDER if t in simulations]
    if not tags:
        logger.error("None of the expected profile tags found. Exiting.")
        return

    # Print mapping
    print("\n--- Profile Mapping ---")
    for data_path in find_all_simulations(Path(args.folder)):
        folder_name = data_path.parent.name
        tag = identify_profile_tag(folder_name)
        print(f"  {folder_name}  ->  {tag or '(unrecognised)'}")
    print()

    # ------------------------------------------------------------------
    # Extract time-series and scalars
    # ------------------------------------------------------------------
    all_ts = {}
    all_scalars = {}
    for tag in tags:
        ts = extract_timeseries(simulations[tag])
        all_ts[tag] = ts
        all_scalars[tag] = extract_scalars(ts)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print_summary_table(all_scalars, tags)

    # ------------------------------------------------------------------
    # Determine output directory
    # ------------------------------------------------------------------
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        folder_name = Path(args.folder).name
        output_dir = FIG_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plot_comparison(all_ts, tags, output_dir, args.fmt, args.show)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    csv_path = write_csv(all_scalars, tags, output_dir)
    print(f"CSV: {csv_path}")


if __name__ == '__main__':
    main()
