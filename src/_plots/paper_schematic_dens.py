#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Composite radial n(r) & T(r) schematic figure for the TRINITY paper.

Splices an early-time bubble profile (resolved T(r), n(r) from the
energy-driven phase) with a late-time shell structure (ionized + neutral
layers) from the same simulation to create a single seamless radial
structure diagram.

Usage:
  python paper_schematic_dens.py -F /path/to/simulations/
  python paper_schematic_dens.py -F /path/to/simulations/ --model 5e5_sfe030_n1e4_PL0
  python paper_schematic_dens.py -F /path/to/simulations/ --save-dir fig/

@author: Jia Wei Teh
"""

import sys
import os
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import (
    load_output, find_all_simulations, parse_simulation_params,
)
from src._functions.unit_conversions import CONV, INV_CONV, CGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Load matplotlib style
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))

# Output figure directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Physical constants (not in snapshot — use standard values)
# =============================================================================
T_ION = 1e4      # K — ionized gas (H II region)
T_NEU = 100.0    # K — neutral shell (PDR + cold)
T_CLOUD = 15.0   # K — ambient molecular cloud
T_ISM = 1e4      # K — warm ISM
N_ISM_CGS = 0.1  # cm^-3 — ISM number density

# Conversion factor: code density (1/pc^3) -> physical (cm^-3)
NDENS_AU2CGS = INV_CONV.ndens_au2cgs


# =============================================================================
# Step 1: Find a suitable simulation + shell snapshot
# =============================================================================

def find_shell_snapshot(output):
    """
    Search all snapshots for one with a partially ionized shell.

    Selection criteria (in order of preference):
      1. Partially ionized: shell_ion_idx > 0, != -1,
         and < len(shell_n_arr) - 1
      2. Fallback: fully ionized (shell_ion_idx == len - 1)

    Parameters
    ----------
    output : TrinityOutput
        Loaded simulation output.

    Returns
    -------
    tuple (snap_index, snap_dict, is_partial)
        Index into output, the snapshot dict, and whether ionization
        is partial (True) or fully ionized fallback (False).
        Returns (None, None, False) if nothing usable found.
    """
    best_full_idx = None
    best_full_snap = None

    for i in range(len(output)):
        snap = output[i]

        # Read shell arrays
        shell_r = snap.get('shell_r_arr', None)
        log_shell_n = snap.get('log_shell_n_arr', None)
        ion_idx = snap.get('shell_ion_idx', None)

        # Must have arrays
        if shell_r is None or log_shell_n is None:
            continue
        shell_r = np.asarray(shell_r, dtype=float)
        log_shell_n = np.asarray(log_shell_n, dtype=float)
        if shell_r.size == 0 or log_shell_n.size == 0:
            continue

        if ion_idx is None:
            continue
        ion_idx = int(ion_idx)

        # Determine R_IF for this snapshot to locate boundary in simplified arrays
        R_IF = snap.get('R_IF', None)

        # -1 means photons escaped — skip for partial criterion
        if ion_idx == -1:
            continue

        # Count original array length (shell_ion_idx is from the original,
        # but we can still use it to classify partial vs. fully ionized)
        n_arr_len = shell_r.size  # simplified array length (good enough)

        # Fully ionized (all photons absorbed within shell)
        if ion_idx >= n_arr_len - 1:
            if best_full_idx is None:
                best_full_idx = i
                best_full_snap = snap
            continue

        # Partially ionized — we need ion_idx > 0
        if ion_idx > 0 and R_IF is not None and float(R_IF) > 0:
            return i, snap, True

    # Fallback: fully ionized
    if best_full_snap is not None:
        return best_full_idx, best_full_snap, False

    return None, None, False


# =============================================================================
# Step 2: Find an early-time bubble snapshot
# =============================================================================

def find_bubble_snapshot(output):
    """
    Find a snapshot with resolved bubble T(r), n(r) arrays.

    Prefers the latest snapshot that still has Eb > 0 and
    populated bubble arrays (length > 10).

    Parameters
    ----------
    output : TrinityOutput

    Returns
    -------
    tuple (snap_index, snap_dict) or (None, None)
    """
    best_idx = None
    best_snap = None

    for i in range(len(output)):
        snap = output[i]

        Eb = snap.get('Eb', 0)
        if Eb is None or float(Eb) <= 0:
            continue

        log_T = snap.get('log_bubble_T_arr', None)
        log_n = snap.get('log_bubble_n_arr', None)
        r_T = snap.get('bubble_T_arr_r_arr', None)

        if log_T is None or log_n is None or r_T is None:
            continue

        log_T = np.asarray(log_T, dtype=float)
        log_n = np.asarray(log_n, dtype=float)
        r_T = np.asarray(r_T, dtype=float)

        if log_T.size <= 10 or log_n.size <= 10 or r_T.size == 0:
            continue

        # Valid — keep updating to get the latest (closest to transition)
        best_idx = i
        best_snap = snap

    return best_idx, best_snap


# =============================================================================
# Step 3 & 4: Build the composite radial profile (with rescaling)
# =============================================================================

def build_composite(shell_snap, bubble_snap, is_partial, nCore_cgs):
    """
    Build composite arrays of (r, n, T) across all six radial zones.

    Returns
    -------
    dict with keys:
        'zones'  : list of dicts, each with 'r', 'n', 'T', 'label', 'color'
        'boundaries' : dict  {name: r_value} for vertical lines
    """
    # ------------------------------------------------------------------
    # Reference frame: shell snapshot radii
    # ------------------------------------------------------------------
    R2_shell = float(shell_snap.get('R2', 0))
    rShell_shell = float(shell_snap.get('rShell', 0))
    rCloud = float(shell_snap.get('rCloud', 1.0))
    R_IF_shell = float(shell_snap.get('R_IF', R2_shell))

    # Shell arrays (simplified, from serialization)
    shell_r = np.asarray(shell_snap.get('shell_r_arr', []), dtype=float)
    log_shell_n = np.asarray(shell_snap.get('log_shell_n_arr', []), dtype=float)

    # Convert shell density: code (1/pc^3) -> physical (cm^-3)
    shell_n_cgs = 10.0 ** log_shell_n * NDENS_AU2CGS

    # Locate ionization front in the simplified array using R_IF
    if is_partial and R_IF_shell > 0:
        ion_mask = shell_r <= R_IF_shell
        if np.any(ion_mask):
            ion_idx_simplified = np.max(np.where(ion_mask))
        else:
            ion_idx_simplified = 0
    else:
        # Fully ionized — whole array is ionized
        ion_idx_simplified = len(shell_r) - 1

    # ------------------------------------------------------------------
    # Bubble snapshot values
    # ------------------------------------------------------------------
    R1_bub = float(bubble_snap.get('R1', 0))
    R2_bub = float(bubble_snap.get('R2', 0))

    # Bubble profile arrays
    bub_r_T = np.asarray(bubble_snap.get('bubble_T_arr_r_arr', []), dtype=float)
    log_bub_T = np.asarray(bubble_snap.get('log_bubble_T_arr', []), dtype=float)
    bub_r_n = np.asarray(bubble_snap.get('bubble_n_arr_r_arr', []), dtype=float)
    log_bub_n = np.asarray(bubble_snap.get('log_bubble_n_arr', []), dtype=float)

    # Sort both bubble profiles by ascending r (they come R2 -> R1)
    if bub_r_T.size > 1 and bub_r_T[0] > bub_r_T[-1]:
        order_T = np.argsort(bub_r_T)
        bub_r_T = bub_r_T[order_T]
        log_bub_T = log_bub_T[order_T]
    if bub_r_n.size > 1 and bub_r_n[0] > bub_r_n[-1]:
        order_n = np.argsort(bub_r_n)
        bub_r_n = bub_r_n[order_n]
        log_bub_n = log_bub_n[order_n]

    bub_T = 10.0 ** log_bub_T          # K
    bub_n_cgs = 10.0 ** log_bub_n * NDENS_AU2CGS  # cm^-3

    # ------------------------------------------------------------------
    # Step 4: Rescale bubble radii into the shell snapshot reference frame
    # ------------------------------------------------------------------
    # Maintain the physical ratio R1/R2 from the bubble snapshot
    ratio_R1R2 = R1_bub / R2_bub if R2_bub > 0 else 0.1
    R1_rescaled = ratio_R1R2 * R2_shell

    # Linearly map [R1_bub, R2_bub] -> [R1_rescaled, R2_shell]
    if R2_bub > R1_bub:
        scale = (R2_shell - R1_rescaled) / (R2_bub - R1_bub)
        bub_r_T_rescaled = R1_rescaled + (bub_r_T - R1_bub) * scale
        bub_r_n_rescaled = R1_rescaled + (bub_r_n - R1_bub) * scale
    else:
        bub_r_T_rescaled = bub_r_T
        bub_r_n_rescaled = bub_r_n

    # ------------------------------------------------------------------
    # Zone 1: Free-streaming wind (0.01 pc -> R1)
    # ------------------------------------------------------------------
    pdot_W = float(bubble_snap.get('pdot_W', 0))
    Lmech_W = float(bubble_snap.get('Lmech_W', 0))

    # Ensure R1_rescaled is large enough to display a wind zone
    R1_rescaled = max(R1_rescaled, R2_shell * 0.02)

    r_wind_min = R1_rescaled * 0.01
    r_wind = np.logspace(np.log10(r_wind_min), np.log10(R1_rescaled), 50,
                         endpoint=False)  # exclude R1 itself

    if pdot_W > 0 and Lmech_W > 0:
        # v_w = 2 * Lmech / pdot  [pc/Myr]
        v_w = 2.0 * Lmech_W / pdot_W
        # Mdot_w = pdot / v_w  [Msun/Myr]
        Mdot_w = pdot_W / v_w
        # n(r) = Mdot / (4 pi r^2 v_w mu m_H)  in code units (1/pc^3)
        # mu_ion ~ 0.6 (ionized), m_H in Msun
        mu_ion = 0.6
        m_H_Msun = CGS.m_H * CONV.g2Msun
        n_wind_code = Mdot_w / (4.0 * np.pi * r_wind ** 2 * v_w * mu_ion * m_H_Msun)
        n_wind_cgs = n_wind_code * NDENS_AU2CGS
    else:
        # Fallback: extrapolate from inner edge of bubble
        n_inner = bub_n_cgs[0] if bub_n_cgs.size > 0 else 0.01
        r_inner = bub_r_n_rescaled[0] if bub_r_n_rescaled.size > 0 else R1_rescaled
        n_wind_cgs = n_inner * (r_inner / r_wind) ** 2

    T_wind = np.full_like(r_wind, T_ION)

    # ------------------------------------------------------------------
    # Zone 2: Hot bubble (R1 -> R2)
    # Clip rescaled bubble arrays strictly to [R1_rescaled, R2_shell]
    # ------------------------------------------------------------------
    def _clip_zone(r_arr, y_arr, r_lo, r_hi):
        """Keep only points with r_lo <= r <= r_hi."""
        mask = (r_arr >= r_lo) & (r_arr <= r_hi)
        return r_arr[mask], y_arr[mask]

    r_bubble, n_bubble = _clip_zone(
        bub_r_n_rescaled, bub_n_cgs, R1_rescaled, R2_shell)
    r_bubble_T, T_bubble = _clip_zone(
        bub_r_T_rescaled, bub_T, R1_rescaled, R2_shell)

    # ------------------------------------------------------------------
    # Zone 3: Ionized shell (R2 -> R_IF)
    # ------------------------------------------------------------------
    ion_end = ion_idx_simplified + 1
    r_ion = shell_r[:ion_end]
    n_ion = shell_n_cgs[:ion_end]
    T_ion = np.full_like(r_ion, T_ION)

    # ------------------------------------------------------------------
    # Zone 4: Neutral shell (R_IF -> rShell)
    # ------------------------------------------------------------------
    if is_partial and ion_end < len(shell_r):
        r_neu = shell_r[ion_end:]
        n_neu = shell_n_cgs[ion_end:]
        T_neu_val = float(shell_snap.get('TShell_neu', T_NEU))
        if T_neu_val == 0:
            T_neu_val = T_NEU
        T_neu = np.full_like(r_neu, T_neu_val)
    else:
        r_neu = np.array([])
        n_neu = np.array([])
        T_neu = np.array([])

    # ------------------------------------------------------------------
    # Zone 5: Ambient cloud (rShell -> rCloud)
    # ------------------------------------------------------------------
    if rShell_shell < rCloud:
        r_cloud = np.logspace(
            np.log10(max(rShell_shell, 0.01)),
            np.log10(rCloud),
            50,
        )
        # Uniform (alpha=0) approximation for schematic
        n_cloud_cgs = np.full_like(r_cloud, nCore_cgs)
        T_cloud = np.full_like(r_cloud, T_CLOUD)
    else:
        r_cloud = np.array([])
        n_cloud_cgs = np.array([])
        T_cloud = np.array([])

    # ------------------------------------------------------------------
    # Zone 6: ISM (rCloud -> 2 * rCloud)
    # ------------------------------------------------------------------
    r_ism = np.linspace(rCloud, 2.0 * rCloud, 10)
    n_ism_cgs = np.full_like(r_ism, N_ISM_CGS)
    T_ism = np.full_like(r_ism, T_ISM)

    # ------------------------------------------------------------------
    # Zone boundary radii (ordered inside-out)
    # ------------------------------------------------------------------
    boundaries = {
        r'$R_1$': R1_rescaled,
        r'$R_2$': R2_shell,
        r'$R_{\rm IF}$': R_IF_shell if is_partial else None,
        r'$r_{\rm Shell}$': rShell_shell,
        r'$r_{\rm Cloud}$': rCloud,
    }

    # Build zone list with explicit radial bounds for shading
    zones = [
        {'r': r_wind, 'n': n_wind_cgs, 'T': T_wind,
         'label': 'Free wind', 'color': '#E8F0FE',
         'r_lo': r_wind_min, 'r_hi': R1_rescaled},
        {'r_n': r_bubble, 'n': n_bubble,
         'r_T': r_bubble_T, 'T': T_bubble,
         'label': 'Hot bubble', 'color': '#FFF3E0',
         'r_lo': R1_rescaled, 'r_hi': R2_shell},
        {'r': r_ion, 'n': n_ion, 'T': T_ion,
         'label': r'H\,{\sc ii}', 'color': '#E8F5E9',
         'r_lo': R2_shell, 'r_hi': R_IF_shell},
        {'r': r_neu, 'n': n_neu, 'T': T_neu,
         'label': 'Neutral shell', 'color': '#FBE9E7',
         'r_lo': R_IF_shell, 'r_hi': rShell_shell},
        {'r': r_cloud, 'n': n_cloud_cgs, 'T': T_cloud,
         'label': 'Cloud', 'color': '#F3E5F5',
         'r_lo': rShell_shell, 'r_hi': rCloud},
        {'r': r_ism, 'n': n_ism_cgs, 'T': T_ism,
         'label': 'ISM', 'color': '#E0F7FA',
         'r_lo': rCloud, 'r_hi': 2.0 * rCloud},
    ]

    return zones, boundaries


# =============================================================================
# Step 5 & 6: Plot and save
# =============================================================================

def plot_schematic(zones, boundaries, save_dir, folder_name):
    """
    Create the dual y-axis n(r) & T(r) schematic figure.
    """
    fig, ax_n = plt.subplots(figsize=(8, 4.5))
    ax_T = ax_n.twinx()

    # ------------------------------------------------------------------
    # Zone background shading (use explicit r_lo/r_hi, not data extents)
    # ------------------------------------------------------------------
    for zone in zones:
        r_lo = zone.get('r_lo', None)
        r_hi = zone.get('r_hi', None)
        if r_lo is None or r_hi is None or r_hi <= r_lo:
            continue
        # Skip zones with no data
        r_arr = zone.get('r', zone.get('r_n', np.array([])))
        if r_arr.size == 0:
            continue
        ax_n.axvspan(r_lo, r_hi, color=zone['color'], alpha=0.35, zorder=0)

    # ------------------------------------------------------------------
    # Density curve (solid, dark blue)
    # ------------------------------------------------------------------
    for zone in zones:
        # Zone 2 (bubble) has separate r for n and T
        r_n = zone.get('r', zone.get('r_n', np.array([])))
        n_arr = zone.get('n', np.array([]))
        if r_n.size == 0 or n_arr.size == 0:
            continue
        # Ensure matching sizes (simplified arrays may differ slightly)
        n_plot = min(r_n.size, n_arr.size)
        mask = n_arr[:n_plot] > 0
        if np.any(mask):
            ax_n.plot(
                r_n[:n_plot][mask], n_arr[:n_plot][mask],
                color='#0072B2', lw=2.0, solid_capstyle='round', zorder=5,
            )

    # ------------------------------------------------------------------
    # Temperature curve (dashed, red-orange)
    # ------------------------------------------------------------------
    for zone in zones:
        r_T = zone.get('r', zone.get('r_T', np.array([])))
        T_arr = zone.get('T', np.array([]))
        if r_T.size == 0 or T_arr.size == 0:
            continue
        n_plot = min(r_T.size, T_arr.size)
        mask = T_arr[:n_plot] > 0
        if np.any(mask):
            ax_T.plot(
                r_T[:n_plot][mask], T_arr[:n_plot][mask],
                color='#D55E00', lw=2.0, ls='--', solid_capstyle='round',
                zorder=5,
            )

    # ------------------------------------------------------------------
    # Zone boundary lines
    # ------------------------------------------------------------------
    for label, r_val in boundaries.items():
        if r_val is None or r_val <= 0:
            continue
        ax_n.axvline(r_val, color='0.4', lw=0.7, ls=':', zorder=3)
        # Place label at top of axes
        ax_n.text(
            r_val, 1.02, label,
            transform=ax_n.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=10, color='0.3',
        )

    # ------------------------------------------------------------------
    # Zone name labels (positioned at geometric centre of zone bounds)
    # ------------------------------------------------------------------
    for zone in zones:
        r_lo = zone.get('r_lo', None)
        r_hi = zone.get('r_hi', None)
        r_arr = zone.get('r', zone.get('r_n', np.array([])))
        if r_lo is None or r_hi is None or r_hi <= r_lo or r_arr.size == 0:
            continue
        r_mid = np.sqrt(r_lo * r_hi)  # geometric mean of zone bounds
        ax_n.text(
            r_mid, 0.04, zone['label'],
            transform=ax_n.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=8, color='0.35',
            style='italic',
        )

    # ------------------------------------------------------------------
    # Axes formatting
    # ------------------------------------------------------------------
    ax_n.set_xscale('log')
    ax_n.set_yscale('log')
    ax_T.set_yscale('log')

    ax_n.set_xlabel(r'$r$ [pc]')
    ax_n.set_ylabel(r'$n\;[\mathrm{cm}^{-3}]$', color='#0072B2')
    ax_T.set_ylabel(r'$T\;[\mathrm{K}]$', color='#D55E00')

    ax_n.tick_params(axis='y', colors='#0072B2')
    ax_T.tick_params(axis='y', colors='#D55E00')

    # Collect all plotted r values for xlim
    all_r = []
    for zone in zones:
        r_arr = zone.get('r', zone.get('r_n', np.array([])))
        if r_arr.size > 0:
            all_r.append(r_arr)
    if all_r:
        r_all = np.concatenate(all_r)
        r_all = r_all[r_all > 0]
        if r_all.size > 0:
            ax_n.set_xlim(r_all.min() * 0.5, r_all.max() * 1.5)

    # Annotation: "Schematic"
    ax_n.text(
        0.98, 0.96, 'Schematic',
        transform=ax_n.transAxes,
        ha='right', va='top', fontsize=11,
        fontstyle='italic', color='0.4',
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#0072B2', lw=2.0, label=r'$n(r)$'),
        Line2D([0], [0], color='#D55E00', lw=2.0, ls='--', label=r'$T(r)$'),
    ]
    ax_n.legend(handles=legend_elements, loc='upper right', framealpha=0.8)

    fig.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out_pdf = save_dir / 'paper_schematic_dens.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    logger.info(f"Saved: {out_pdf}")

    out_png = save_dir / 'paper_schematic_dens.png'
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    logger.info(f"Saved: {out_png}")

    plt.close(fig)
    return out_pdf, out_png


# =============================================================================
# Fallback: analytic bubble profile (Weaver 1977 similarity solution)
# =============================================================================

def analytic_bubble(R1, R2, n_pts=60):
    """
    Weaver (1977) similarity solution for the hot bubble interior.

    Returns r, n (cm^-3), T (K) arrays spanning [R1, R2].
    """
    r = np.linspace(R1, R2, n_pts)
    x = (r - R1) / (R2 - R1)  # normalized 0 -> 1

    # T(x) ~ T_max * (1 - x^10)^(2/5)  (Weaver+77 Eq. 34)
    T_max = 3e7  # K (typical peak)
    T = T_max * np.maximum(1.0 - x ** 10, 1e-6) ** 0.4

    # Pressure balance: n * k_B * T = const => n ∝ 1/T
    P_over_kB = 1e7  # K cm^-3 (typical)
    n = P_over_kB / T

    return r, n, T


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TRINITY radial n(r) & T(r) schematic figure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_schematic_dens.py -F /path/to/simulations/
  python paper_schematic_dens.py -F /path/to/simulations/ --model 5e5_sfe030_n1e4_PL0
  python paper_schematic_dens.py -F /path/to/simulations/ --save-dir fig/schematic/
        """,
    )
    parser.add_argument(
        '--folder', '-F', required=True,
        help='Path to parent folder containing simulation subfolders',
    )
    parser.add_argument(
        '--model', default=None,
        help='Specific subfolder / model name (e.g. 5e5_sfe030_n1e4_PL0)',
    )
    parser.add_argument(
        '--save-dir', default=None,
        help='Directory to save figures (default: same as -F)',
    )

    args = parser.parse_args()
    folder = Path(args.folder)

    # ------------------------------------------------------------------
    # Discover simulations
    # ------------------------------------------------------------------
    sim_files = find_all_simulations(folder)
    if not sim_files:
        logger.error(f"No simulations found in {folder}")
        return

    logger.info(f"Found {len(sim_files)} simulation(s) in {folder}")

    # If --model given, filter to that one
    if args.model:
        sim_files = [p for p in sim_files if p.parent.name == args.model]
        if not sim_files:
            logger.error(f"Model '{args.model}' not found in {folder}")
            return

    # ------------------------------------------------------------------
    # Search for a suitable simulation
    # ------------------------------------------------------------------
    chosen_path = None
    shell_snap = None
    bubble_snap = None
    is_partial = False
    nCore_cgs = 1e4  # default

    for data_path in sim_files:
        folder_name = data_path.parent.name
        logger.info(f"Checking {folder_name} ...")

        output = load_output(data_path)
        if len(output) == 0:
            logger.info(f"  Skipping {folder_name}: empty output")
            continue

        # Step 1: look for shell snapshot
        sh_idx, sh_snap, partial = find_shell_snapshot(output)
        if sh_snap is None:
            logger.info(f"  Skipping {folder_name}: no usable shell snapshot")
            continue

        # Step 2: look for bubble snapshot
        bub_idx, bub_snap = find_bubble_snapshot(output)
        if bub_snap is None:
            logger.info(f"  Skipping {folder_name}: no usable bubble snapshot")
            continue

        # Success — use this simulation
        chosen_path = data_path
        shell_snap = sh_snap
        bubble_snap = bub_snap
        is_partial = partial

        # Extract nCore from folder name
        parsed = parse_simulation_params(folder_name)
        if parsed and 'ndens' in parsed:
            nCore_cgs = float(parsed['ndens'])

        if partial:
            logger.info(
                f"  Selected {folder_name}: partial ionization "
                f"(shell snap #{sh_idx}, bubble snap #{bub_idx})"
            )
        else:
            logger.warning(
                f"  Selected {folder_name}: fully ionized fallback "
                f"(no neutral layer available)"
            )
        break

    if chosen_path is None:
        logger.error("No suitable simulation found across all subfolders.")
        return

    # ------------------------------------------------------------------
    # Build the composite profile
    # ------------------------------------------------------------------
    zones, boundaries = build_composite(
        shell_snap, bubble_snap, is_partial, nCore_cgs,
    )

    # If bubble arrays were empty, use analytic fallback
    bub_zone = zones[1]  # Hot bubble
    r_n = bub_zone.get('r_n', np.array([]))
    if r_n.size == 0:
        logger.warning("Bubble arrays empty — using Weaver (1977) analytic profile")
        R1 = boundaries[r'$R_1$']
        R2 = boundaries[r'$R_2$']
        r_a, n_a, T_a = analytic_bubble(R1, R2)
        zones[1] = {
            'r_n': r_a, 'n': n_a,
            'r_T': r_a, 'T': T_a,
            'label': 'Hot bubble', 'color': '#FFF3E0',
        }

    # ------------------------------------------------------------------
    # Plot and save
    # ------------------------------------------------------------------
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = FIG_DIR / folder.name

    model_name = chosen_path.parent.name
    pdf_path, png_path = plot_schematic(zones, boundaries, save_dir, model_name)

    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")


if __name__ == '__main__':
    main()
