#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radial density profile builder for TRINITY.

Constructs a seamless n(r) profile from centre to ISM for any snapshot,
covering all six radial zones:

    1. Free-streaming wind  (0       → R1)
    2. Hot bubble           (R1      → R2)
    3. Ionised shell        (R2      → R_IF)
    4. Neutral shell        (R_IF    → rShell)
    5. Molecular cloud      (rShell  → rCloud)
    6. ISM                  (rCloud  → ∞)

Two main entry points:

    build_radial_profile(snap, ...)
        Build (r, n) for a single snapshot.  Returns a RadialProfile dataclass.

    build_profile_table(output, ...)
        Build a table of (t, r, n) for all usable snapshots in a simulation.

    animate_profiles(output, ...)
        Create an animated .gif of the evolving density profile.

Units
-----
All radii are in **pc**, all densities in **cm⁻³**.

@author: Jia Wei Teh
"""

import logging
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src._functions.unit_conversions import INV_CONV, CGS, CONV

logger = logging.getLogger(__name__)

# Conversion: code density (1/pc³) → physical (cm⁻³)
_NDENS_AU2CGS = INV_CONV.ndens_au2cgs

# Physical temperatures for zone labelling (not used in density, but useful metadata)
T_ION = 1e4      # K — ionised shell
T_NEU = 100.0    # K — neutral shell / PDR
T_CLOUD = 15.0   # K — molecular cloud
T_ISM = 1e4      # K — warm ISM


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class ZoneProfile:
    """Density profile for a single radial zone."""
    name: str            # e.g. 'free_wind', 'hot_bubble', ...
    r: np.ndarray        # radii  [pc]
    n: np.ndarray        # number density [cm⁻³]
    r_lo: float          # zone inner boundary [pc]
    r_hi: float          # zone outer boundary [pc]


@dataclass
class RadialProfile:
    """Complete radial density profile for one snapshot."""
    t: float                       # simulation time [Myr]
    phase: str                     # simulation phase
    zones: List[ZoneProfile]       # ordered list of zone profiles
    r: np.ndarray = field(init=False, repr=False)  # concatenated r [pc]
    n: np.ndarray = field(init=False, repr=False)  # concatenated n [cm⁻³]

    def __post_init__(self):
        parts_r, parts_n = [], []
        for z in self.zones:
            if z.r.size > 0:
                parts_r.append(z.r)
                parts_n.append(z.n)
        if parts_r:
            self.r = np.concatenate(parts_r)
            self.n = np.concatenate(parts_n)
        else:
            self.r = np.array([])
            self.n = np.array([])


# =============================================================================
# Zone builders
# =============================================================================

def _build_free_wind(snap, R1: float, n_pts: int = 50) -> ZoneProfile:
    """
    Zone 1: Free-streaming wind.

    n(r) = Mdot_w / (4π r² v_w μ m_H)  ∝ r⁻²

    Uses saved scalars pdot_W and Lmech_W to reconstruct.
    """
    pdot_W = float(snap.get('pdot_W', 0) or 0)
    Lmech_W = float(snap.get('Lmech_W', 0) or 0)

    r_min = R1 * 0.01
    r_wind = np.logspace(np.log10(max(r_min, 1e-6)), np.log10(R1), n_pts,
                         endpoint=False)

    if pdot_W > 0 and Lmech_W > 0:
        v_w = 2.0 * Lmech_W / pdot_W          # pc/Myr
        Mdot_w = pdot_W / v_w                  # Msun/Myr
        mu_ion = 0.6
        m_H_Msun = CGS.m_H * CONV.g2Msun
        n_code = Mdot_w / (4.0 * np.pi * r_wind**2 * v_w * mu_ion * m_H_Msun)
        n_cgs = n_code * _NDENS_AU2CGS
    else:
        # Fallback: flat low density
        n_cgs = np.full_like(r_wind, 1e-3)

    return ZoneProfile('free_wind', r_wind, n_cgs, r_min, R1)


def _build_hot_bubble(snap, R1: float, R2: float) -> ZoneProfile:
    """
    Zone 2: Hot bubble from saved arrays.

    Available only in energy / implicit / transition phases (Eb > 0).
    """
    bub_r = np.asarray(snap.get('bubble_n_arr_r_arr', []) or [], dtype=float)
    log_bub_n = np.asarray(snap.get('log_bubble_n_arr', []) or [], dtype=float)

    if bub_r.size == 0 or log_bub_n.size == 0:
        return ZoneProfile('hot_bubble', np.array([]), np.array([]), R1, R2)

    # Sort ascending by radius (arrays come R2 → R1)
    if bub_r.size > 1 and bub_r[0] > bub_r[-1]:
        order = np.argsort(bub_r)
        bub_r = bub_r[order]
        log_bub_n = log_bub_n[order]

    # Clip to [R1, R2]
    mask = (bub_r >= R1) & (bub_r <= R2)
    bub_r = bub_r[mask]
    log_bub_n = log_bub_n[mask]

    n_cgs = 10.0**log_bub_n * _NDENS_AU2CGS

    return ZoneProfile('hot_bubble', bub_r, n_cgs, R1, R2)


def _build_bubble_interpolated(snap, R1: float, R2: float,
                               n_wind_at_R1: float,
                               n_shell_at_R2: float,
                               n_pts: int = 30) -> ZoneProfile:
    """
    Zone 2 fallback: interpolate between wind endpoint and shell inner edge.

    Used during transition / momentum phase when bubble arrays are absent.
    Log-linear interpolation in (r, log n).
    """
    if R2 <= R1 or R1 <= 0:
        return ZoneProfile('hot_bubble', np.array([]), np.array([]), R1, R2)

    r_arr = np.logspace(np.log10(R1), np.log10(R2), n_pts)

    # Protect against zero/negative densities
    n_lo = max(n_wind_at_R1, 1e-6)
    n_hi = max(n_shell_at_R2, 1e-6)

    log_n = np.interp(np.log10(r_arr),
                      [np.log10(R1), np.log10(R2)],
                      [np.log10(n_lo), np.log10(n_hi)])
    n_cgs = 10.0**log_n

    return ZoneProfile('hot_bubble', r_arr, n_cgs, R1, R2)


def _build_ionised_shell(snap, shell_r: np.ndarray, shell_n_cgs: np.ndarray,
                         R2: float, R_IF: float,
                         ion_idx: int) -> ZoneProfile:
    """Zone 3: Ionised shell (R2 → R_IF)."""
    if shell_r.size == 0 or ion_idx < 0:
        return ZoneProfile('ionised_shell', np.array([]), np.array([]), R2, R_IF)

    # Use R_IF to locate boundary in the (possibly simplified) arrays
    if R_IF > 0:
        ion_mask = shell_r <= R_IF
        if np.any(ion_mask):
            end = np.max(np.where(ion_mask)) + 1
        else:
            end = 1
    else:
        end = min(ion_idx + 1, len(shell_r))

    r_ion = shell_r[:end]
    n_ion = shell_n_cgs[:end]

    r_hi = R_IF if R_IF > 0 else (r_ion[-1] if r_ion.size > 0 else R2)
    return ZoneProfile('ionised_shell', r_ion, n_ion, R2, r_hi)


def _build_neutral_shell(snap, shell_r: np.ndarray, shell_n_cgs: np.ndarray,
                         R_IF: float, rShell: float,
                         ion_idx: int) -> ZoneProfile:
    """Zone 4: Neutral shell (R_IF → rShell)."""
    if shell_r.size == 0 or ion_idx < 0:
        return ZoneProfile('neutral_shell', np.array([]), np.array([]), R_IF, rShell)

    # Locate where neutral region starts
    if R_IF > 0:
        neu_mask = shell_r > R_IF
        if np.any(neu_mask):
            start = np.min(np.where(neu_mask))
        else:
            return ZoneProfile('neutral_shell', np.array([]), np.array([]),
                               R_IF, rShell)
    else:
        start = min(ion_idx + 1, len(shell_r))

    if start >= len(shell_r):
        return ZoneProfile('neutral_shell', np.array([]), np.array([]),
                           R_IF, rShell)

    r_neu = shell_r[start:]
    n_neu = shell_n_cgs[start:]
    return ZoneProfile('neutral_shell', r_neu, n_neu, R_IF, rShell)


def _build_cloud(snap, rShell: float, rCloud: float,
                 n_pts: int = 80) -> ZoneProfile:
    """
    Zone 5: Molecular cloud (rShell → rCloud).

    Uses the saved initial_cloud arrays to interpolate the actual cloud profile.
    Falls back to a constant nCore if arrays are unavailable.
    """
    if rShell >= rCloud:
        return ZoneProfile('cloud', np.array([]), np.array([]), rShell, rCloud)

    cloud_r = np.asarray(snap.get('initial_cloud_r_arr', []) or [], dtype=float)
    cloud_n = np.asarray(snap.get('initial_cloud_n_arr', []) or [], dtype=float)

    r_arr = np.logspace(np.log10(max(rShell, 1e-4)), np.log10(rCloud), n_pts)

    if cloud_r.size > 1 and cloud_n.size > 1:
        # Cloud density arrays are already in cm⁻³
        n_interp = np.interp(r_arr, cloud_r, cloud_n)
    else:
        # Fallback: use nCore if available
        nCore = snap.get('nCore', None)
        if nCore is not None and float(nCore) > 0:
            n_interp = np.full_like(r_arr, float(nCore))
        else:
            # Last resort: extrapolate from nEdge
            nEdge = float(snap.get('nEdge', 100) or 100)
            n_interp = np.full_like(r_arr, nEdge)

    return ZoneProfile('cloud', r_arr, n_interp, rShell, rCloud)


def _build_ism(snap, rCloud: float, n_pts: int = 10) -> ZoneProfile:
    """Zone 6: ISM (rCloud → 2·rCloud)."""
    nISM = snap.get('nISM', None)
    if nISM is not None and float(nISM) > 0:
        n_val = float(nISM)
    else:
        n_val = 0.1  # default ISM density [cm⁻³]

    r_arr = np.linspace(rCloud, 2.0 * rCloud, n_pts)
    n_arr = np.full_like(r_arr, n_val)
    return ZoneProfile('ism', r_arr, n_arr, rCloud, 2.0 * rCloud)


# =============================================================================
# Main builder
# =============================================================================

def build_radial_profile(snap) -> Optional[RadialProfile]:
    """
    Build the complete radial density profile for a single snapshot.

    Parameters
    ----------
    snap : Snapshot or dict-like
        A single TRINITY snapshot (from TrinityOutput[i] or output.get_at_time()).

    Returns
    -------
    RadialProfile or None
        The composite density profile, or None if the snapshot lacks shell data.
    """
    # --- Read key radii ---
    R1 = float(snap.get('R1', 0) or 0)
    R2 = float(snap.get('R2', 0) or 0)
    rShell = float(snap.get('rShell', 0) or 0)
    rCloud = float(snap.get('rCloud', 1.0) or 1.0)
    R_IF = float(snap.get('R_IF', 0) or 0)
    Eb = float(snap.get('Eb', 0) or 0)
    t_now = float(snap.get('t_now', 0) or 0)
    phase = str(snap.get('current_phase', 'unknown') or 'unknown')
    ion_idx = snap.get('shell_ion_idx', -1)
    if ion_idx is not None:
        ion_idx = int(ion_idx)
    else:
        ion_idx = -1

    # --- Validate: need shell structure ---
    shell_r = np.asarray(snap.get('shell_r_arr', []) or [], dtype=float)
    log_shell_n = np.asarray(snap.get('log_shell_n_arr', []) or [], dtype=float)

    if shell_r.size == 0 or log_shell_n.size == 0:
        return None  # skip snapshots without shell structure

    if R2 <= 0:
        return None

    # Convert shell density: code (1/pc³) → physical (cm⁻³)
    shell_n_cgs = 10.0**log_shell_n * _NDENS_AU2CGS

    # --- Zone 1: Free-streaming wind ---
    # In momentum phase R1 = R2, so wind zone is zero-width. Use a small R1.
    has_bubble = Eb > 0 and R1 > 0 and R1 < R2
    if has_bubble:
        wind_zone = _build_free_wind(snap, R1)
    else:
        # Momentum phase: shrink wind to a tiny zone inside R2
        R1_eff = R2 * 0.01
        wind_zone = _build_free_wind(snap, R1_eff)

    # --- Zone 2: Hot bubble ---
    if has_bubble:
        bubble_zone = _build_hot_bubble(snap, R1, R2)
        # If bubble arrays are empty (e.g. late transition), interpolate
        if bubble_zone.r.size == 0:
            n_wind_R1 = wind_zone.n[-1] if wind_zone.n.size > 0 else 1e-3
            n_shell_R2 = shell_n_cgs[0] if shell_n_cgs.size > 0 else 1.0
            bubble_zone = _build_bubble_interpolated(
                snap, R1, R2, n_wind_R1, n_shell_R2)
    else:
        # Momentum/post-transition: interpolate from wind edge to shell inner
        R1_eff = R2 * 0.01
        n_wind_R1 = wind_zone.n[-1] if wind_zone.n.size > 0 else 1e-3
        n_shell_R2 = shell_n_cgs[0] if shell_n_cgs.size > 0 else 1.0
        bubble_zone = _build_bubble_interpolated(
            snap, R1_eff, R2, n_wind_R1, n_shell_R2)

    # --- Zones 3 & 4: Ionised and neutral shell ---
    ion_zone = _build_ionised_shell(snap, shell_r, shell_n_cgs, R2, R_IF, ion_idx)
    neu_zone = _build_neutral_shell(snap, shell_r, shell_n_cgs, R_IF, rShell, ion_idx)

    # --- Zone 5: Molecular cloud ---
    cloud_zone = _build_cloud(snap, rShell, rCloud)

    # --- Zone 6: ISM ---
    ism_zone = _build_ism(snap, rCloud)

    zones = [wind_zone, bubble_zone, ion_zone, neu_zone, cloud_zone, ism_zone]
    return RadialProfile(t=t_now, phase=phase, zones=zones)


# =============================================================================
# Table builder
# =============================================================================

def build_profile_table(output, max_r_pts: int = 300
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a table of (t, r, n) for all usable snapshots.

    Parameters
    ----------
    output : TrinityOutput
        Loaded simulation output.
    max_r_pts : int
        Maximum number of radial points per snapshot (for uniform grid).

    Returns
    -------
    t_arr : np.ndarray, shape (N_snap,)
        Times [Myr] of usable snapshots.
    r_grid : np.ndarray, shape (N_r,)
        Common radial grid [pc].
    n_table : np.ndarray, shape (N_snap, N_r)
        Density [cm⁻³] interpolated onto the common grid.
        NaN where no data is available.
    """
    # First pass: collect all profiles and determine radial extent
    profiles = []
    for i in range(len(output)):
        prof = build_radial_profile(output[i])
        if prof is not None and prof.r.size > 0:
            profiles.append(prof)

    if not profiles:
        return np.array([]), np.array([]), np.empty((0, 0))

    # Common radial grid spanning all snapshots
    r_min = min(p.r[p.r > 0].min() for p in profiles if np.any(p.r > 0))
    r_max = max(p.r.max() for p in profiles)
    r_grid = np.logspace(np.log10(r_min), np.log10(r_max), max_r_pts)

    t_arr = np.array([p.t for p in profiles])
    n_table = np.full((len(profiles), max_r_pts), np.nan)

    for i, prof in enumerate(profiles):
        # Interpolate onto common grid (log-space)
        valid = (prof.r > 0) & (prof.n > 0)
        if np.sum(valid) < 2:
            continue
        log_n_interp = np.interp(
            np.log10(r_grid),
            np.log10(prof.r[valid]),
            np.log10(prof.n[valid]),
            left=np.nan, right=np.nan,
        )
        n_table[i] = 10.0**log_n_interp

    return t_arr, r_grid, n_table


# =============================================================================
# Animation
# =============================================================================

def animate_profiles(output, save_path: str = 'density_profile.gif',
                     fps: int = 8, dpi: int = 150,
                     figsize: Tuple[float, float] = (9, 5)):
    """
    Create an animated .gif of the evolving radial density profile.

    Parameters
    ----------
    output : TrinityOutput
        Loaded simulation output.
    save_path : str
        Output path for the .gif file.
    fps : int
        Frames per second.
    dpi : int
        Resolution.
    figsize : tuple
        Figure size (width, height) in inches.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Collect usable profiles
    profiles = []
    for i in range(len(output)):
        prof = build_radial_profile(output[i])
        if prof is not None and prof.r.size > 0:
            profiles.append(prof)

    if not profiles:
        logger.warning("No usable snapshots for animation.")
        return

    # Determine global axis limits
    all_r = np.concatenate([p.r[p.r > 0] for p in profiles])
    all_n = np.concatenate([p.n[p.n > 0] for p in profiles])
    r_lo, r_hi = all_r.min() * 0.5, all_r.max() * 2.0
    n_lo, n_hi = all_n.min() * 0.3, all_n.max() * 3.0

    # Zone colours
    ZONE_COLORS = {
        'free_wind': '#E8F0FE',
        'hot_bubble': '#FFF3E0',
        'ionised_shell': '#E8F5E9',
        'neutral_shell': '#FBE9E7',
        'cloud': '#F3E5F5',
        'ism': '#E0F7FA',
    }
    ZONE_LABELS = {
        'free_wind': 'Free wind',
        'hot_bubble': 'Hot bubble',
        'ionised_shell': r'H$\,$II',
        'neutral_shell': 'Neutral',
        'cloud': 'Cloud',
        'ism': 'ISM',
    }

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(r_lo, r_hi)
    ax.set_ylim(n_lo, n_hi)
    ax.set_xlabel(r'$r$ [pc]')
    ax.set_ylabel(r'$n\;[\mathrm{cm}^{-3}]$')

    line, = ax.plot([], [], color='#0072B2', lw=2.0)
    time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes,
                        va='top', fontsize=11)
    phase_text = ax.text(0.02, 0.90, '', transform=ax.transAxes,
                         va='top', fontsize=10, fontstyle='italic', color='0.4')
    # Store shading patches for clearing
    _patches = []

    def _update(frame):
        prof = profiles[frame]
        # Clear old shading
        for p in _patches:
            p.remove()
        _patches.clear()

        # Draw zone shading
        for z in prof.zones:
            if z.r.size == 0 or z.r_hi <= z.r_lo:
                continue
            color = ZONE_COLORS.get(z.name, '#F5F5F5')
            patch = ax.axvspan(z.r_lo, z.r_hi, color=color, alpha=0.35, zorder=0)
            _patches.append(patch)

        # Update density line
        valid = (prof.r > 0) & (prof.n > 0)
        line.set_data(prof.r[valid], prof.n[valid])

        time_text.set_text(f't = {prof.t:.4f} Myr')
        phase_text.set_text(prof.phase)
        return line, time_text, phase_text

    anim = FuncAnimation(fig, _update, frames=len(profiles),
                         blit=False, interval=1000 // fps)
    anim.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    logger.info(f"Animation saved: {save_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface."""
    import argparse
    from pathlib import Path
    from src._output.trinity_reader import load_output, find_all_simulations

    parser = argparse.ArgumentParser(
        description="TRINITY radial density profile builder",
    )
    parser.add_argument('--folder', '-F', required=True,
                        help='Path to simulation output folder')
    parser.add_argument('--model', default=None,
                        help='Specific model subfolder name')
    parser.add_argument('--mode', choices=['gif', 'table'], default='table',
                        help='Output mode: gif (animation) or table (npz)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file path')
    parser.add_argument('--fps', type=int, default=8,
                        help='Frames per second (gif mode)')
    parser.add_argument('--max-r-pts', type=int, default=300,
                        help='Radial grid resolution (table mode)')

    args = parser.parse_args()
    folder = Path(args.folder)

    sim_files = find_all_simulations(folder)
    if args.model:
        sim_files = [p for p in sim_files if p.parent.name == args.model]
    if not sim_files:
        logger.error(f"No simulations found in {folder}")
        return

    data_path = sim_files[0]
    model_name = data_path.parent.name
    output = load_output(data_path)
    logger.info(f"Loaded {len(output)} snapshots from {model_name}")

    if args.mode == 'gif':
        out_path = args.output or f'{model_name}_density.gif'
        animate_profiles(output, save_path=out_path, fps=args.fps)
        print(f"GIF: {out_path}")

    elif args.mode == 'table':
        out_path = args.output or f'{model_name}_density_table.npz'
        t_arr, r_grid, n_table = build_profile_table(
            output, max_r_pts=args.max_r_pts)
        np.savez(out_path, t=t_arr, r=r_grid, n=n_table)
        print(f"Table saved: {out_path}")
        print(f"  {len(t_arr)} snapshots, {len(r_grid)} radial points")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s [%(name)s] %(message)s")
    main()
