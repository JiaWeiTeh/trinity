#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pressure evolution + ζ(t) two-panel figure for each TRINITY simulation run.

Top panel:  Pressure terms vs. time (log y, linear x) in CGS (dyn cm⁻²)
Bottom panel: ζ(t) = R_eq / R_St (log y, same x-axis)

Answers the question: "when does P_HII become the independent driver,
decoupled from the bubble?"
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from src._plots.plot_base import FIG_DIR, smooth_1d
from src._output.trinity_reader import load_output, resolve_data_input
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles
from src._functions.unit_conversions import CGS, CONV, INV_CONV

print("...plotting pressure evolution + zeta")

# ======================================================================
# Configuration
# ======================================================================
SMOOTH_WINDOW = 21
PHASE_CHANGE  = True
SAVE_PDF      = True

# Colors — centralised ChromaPalette
from src._plots.force_colors import C  # noqa: E402

C_PHII  = C.PHII
C_DRIVE = C.DRIVE
C_WIND  = C.WIND
C_RAD   = C.RAD
C_SN    = C.SN
C_GRAV  = C.GRAV

# ======================================================================
# Physics constants (CGS)
# ======================================================================
ALPHA_B    = 2.56e-13     # Case-B recombination coefficient [cm³ s⁻¹] at T=10⁴ K
T_ION      = 1e4          # Ionised gas temperature [K]
MU_ION     = 0.678        # Mean molecular weight (ionised)
MU_NEUTRAL = 1.27         # Mean molecular weight (neutral)

# Isothermal sound speed squared: c_i² = k_B T_ion / (μ_ion m_H)  [cm² s⁻²]
C_I_SQ = CGS.k_B * T_ION / (MU_ION * CGS.m_H)


# ======================================================================
# Data loading
# ======================================================================

def load_run(data_path: Path):
    """
    Load run data and compute pressures in CGS + ζ(t).

    Returns
    -------
    t : array
        Time [Myr].
    R2 : array
        Outer bubble radius [pc].
    phase : array of str
        Phase labels.
    pressures : dict of str → array
        Pressure arrays in CGS (dyn cm⁻²).
        Keys: 'Pb', 'P_HII_Str', 'P_HII_ODE', 'P_ram', 'P_rad', 'P_drive'.
    zeta : array
        ζ(t) = R_eq / R_St.
    rcloud : float
        Cloud radius [pc].
    isCollapse : array
        Collapse flag per snapshot.
    """
    output = load_output(data_path)
    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    t     = output.get('t_now')
    R2    = output.get('R2')
    phase = np.array(output.get('current_phase', as_array=False))

    def get_field(field, default=0.0):
        arr = output.get(field)
        if arr is None or (isinstance(arr, np.ndarray) and np.all(arr == None)):
            return np.full(len(output), default)
        return np.where(arr == None, default, arr).astype(float)

    # --- Raw snapshot fields (all in AU) ---
    Pb_au      = get_field('Pb', 0.0)
    P_HII_au   = get_field('P_HII', 0.0)       # ODE boundary condition value
    P_drive_au = get_field('P_drive', 0.0)
    P_ram_au   = get_field('P_ram', 0.0)
    F_rad_au   = get_field('F_rad', 0.0)        # force, not pressure
    Qi_au      = get_field('Qi', 0.0)           # 1/Myr
    pdot_W_au  = get_field('pdot_W', 0.0)       # Msun pc / Myr²
    nEdge_au   = get_field('nEdge', 0.0)        # 1/pc³
    R2_arr     = get_field('R2', 0.0)           # pc
    R_IF_arr   = get_field('R_IF', 0.0)         # pc

    rcloud     = float(output[0].get('rCloud', np.nan))
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # --- Ensure time increasing ---
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, R2, phase = t[order], R2[order], phase[order]
        Pb_au, P_HII_au = Pb_au[order], P_HII_au[order]
        P_drive_au, P_ram_au = P_drive_au[order], P_ram_au[order]
        F_rad_au = F_rad_au[order]
        Qi_au, pdot_W_au = Qi_au[order], pdot_W_au[order]
        nEdge_au = nEdge_au[order]
        R2_arr, R_IF_arr = R2_arr[order], R_IF_arr[order]
        isCollapse = isCollapse[order]

    # --- Convert pressures to CGS (dyn cm⁻²) ---
    Pb_cgs      = Pb_au      * INV_CONV.Pb_au2cgs
    P_HII_cgs   = P_HII_au   * INV_CONV.Pb_au2cgs  # ODE value
    P_drive_cgs = P_drive_au * INV_CONV.Pb_au2cgs
    P_ram_cgs   = P_ram_au   * INV_CONV.Pb_au2cgs

    # P_rad = F_rad / (4πR²)  [AU force / pc² → AU pressure, then to CGS]
    R2_safe = np.maximum(R2_arr, 1e-30)
    P_rad_au = F_rad_au / (4.0 * np.pi * R2_safe**2)
    P_rad_cgs = P_rad_au * INV_CONV.Pb_au2cgs

    # --- Compute P_HII,Str from Strömgren ionisation balance ---
    # n_IF_Str = sqrt(3 Qi / (4π α_B (R_IF³ − R2³)))  [CGS]
    # P_HII_Str = 2 n_IF_Str k_B T_ion                 [CGS]
    Qi_cgs     = Qi_au * CONV.s2Myr           # AU (1/Myr) → CGS (1/s)
    R_IF_cm    = R_IF_arr * INV_CONV.pc2cm    # pc → cm
    R2_cm      = R2_arr * INV_CONV.pc2cm

    vol_HII = (4.0 / 3.0) * np.pi * np.maximum(R_IF_cm**3 - R2_cm**3, 0.0)
    # Guard: when R_IF ≈ R2, the HII volume is ~0 → n_IF_Str undefined.
    # In that limit, set P_HII_Str = 0 (no independent Strömgren region).
    safe_vol = np.where(vol_HII > 0, vol_HII, np.inf)
    n_IF_Str_sq = 3.0 * Qi_cgs / (ALPHA_B * safe_vol * 3.0)
    # Simplify: n² = Qi / ((4π/3) α_B (R_IF³ − R2³)) — factor already in vol_HII
    n_IF_Str_sq = Qi_cgs / (ALPHA_B * safe_vol)
    n_IF_Str_sq = np.maximum(n_IF_Str_sq, 0.0)
    n_IF_Str = np.sqrt(n_IF_Str_sq)
    P_HII_Str_cgs = 2.0 * n_IF_Str * CGS.k_B * T_ION

    # Suppress where Qi is zero or volume is zero
    invalid = (Qi_cgs <= 0) | (vol_HII <= 0)
    P_HII_Str_cgs[invalid] = 0.0

    # --- Compute ζ(t) = R_eq / R_St ---
    # R_St = (3 Qi / (4π α_B n²))^(1/3)    [cm]
    # R_eq = sqrt(3 ṗ_w / (16π ρ c_i²))    [cm]
    nEdge_cgs   = nEdge_au * INV_CONV.ndens_au2cgs  # 1/pc³ → 1/cm³
    pdot_w_cgs  = pdot_W_au * INV_CONV.pdot_au2cgs  # Msun pc/Myr² → dyn

    R_St = np.where(
        (Qi_cgs > 0) & (nEdge_cgs > 0),
        (3.0 * Qi_cgs / (4.0 * np.pi * ALPHA_B * nEdge_cgs**2))**(1.0 / 3.0),
        np.inf
    )

    rho_cloud = nEdge_cgs * MU_NEUTRAL * CGS.m_H
    R_eq = np.where(
        (pdot_w_cgs > 0) & (rho_cloud > 0),
        # Lancaster+2025 Eq.21 (α_p=1): R_eq² = pdot_w/(4π ρ c_i²)
        np.sqrt(pdot_w_cgs / (4.0 * np.pi * rho_cloud * C_I_SQ)),
        0.0
    )

    zeta = np.where(R_St > 0, R_eq / R_St, np.nan)

    pressures = {
        'Pb':         Pb_cgs,
        'P_HII_Str':  P_HII_Str_cgs,
        'P_HII_ODE':  P_HII_cgs,
        'P_ram':      P_ram_cgs,
        'P_rad':      P_rad_cgs,
        'P_drive':    P_drive_cgs,
    }

    return t, R2, phase, pressures, zeta, rcloud, isCollapse


# ======================================================================
# Plotting helpers
# ======================================================================

def _smooth(y, window):
    """Apply smooth_1d with nan handling."""
    finite = np.isfinite(y)
    if not np.any(finite):
        return y
    y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return smooth_1d(y_clean, window, mode="edge")


def plot_pressureZeta_on_ax(
    ax_top, ax_bot, t, R2, phase, pressures, zeta,
    rcloud=np.nan, isCollapse=None,
    smooth_window=SMOOTH_WINDOW,
    phase_change=True,
    show_rcloud=True,
    show_collapse=True,
):
    """
    Plot pressure evolution (top) and ζ(t) (bottom) on given axes.

    Parameters
    ----------
    ax_top, ax_bot : Axes
        Top (pressure) and bottom (ζ) panels.
    t : array, R2 : array, phase : array
        Time, outer radius, phase labels.
    pressures : dict
        Keys: 'Pb', 'P_HII_Str', 'P_HII_ODE', 'P_ram', 'P_rad', 'P_drive'.
    zeta : array
        ζ(t).
    """
    sw = smooth_window

    # --- Add markers on BOTH axes ---
    for ax in (ax_top, ax_bot):
        add_plot_markers(
            ax, t,
            phase=phase if phase_change else None,
            R2=R2 if show_rcloud else None,
            rcloud=rcloud if show_rcloud else None,
            isCollapse=isCollapse if show_collapse else None,
            show_phase=phase_change,
            show_rcloud=show_rcloud,
            show_collapse=show_collapse,
            show_labels=(ax is ax_top),  # labels only on top
            show_momentum_labels=(True if ax is ax_top else False),
        )

    # ================================================================
    # TOP PANEL: Pressures
    # ================================================================
    Pb         = _smooth(pressures['Pb'],         sw)
    P_HII_Str  = _smooth(pressures['P_HII_Str'],  sw)
    P_HII_ODE  = _smooth(pressures['P_HII_ODE'],  sw)
    P_ram      = _smooth(pressures['P_ram'],       sw)
    P_rad      = _smooth(pressures['P_rad'],       sw)
    P_drive    = _smooth(pressures['P_drive'],     sw)

    # Replace zeros with nan for log-scale plotting
    def pos(arr):
        return np.where(arr > 0, arr, np.nan)

    ax_top.plot(t, pos(Pb),        color=C_DRIVE, lw=1.4, ls='-',  alpha=0.9,
                label=r'$P_b$', zorder=3)
    ax_top.plot(t, pos(P_HII_Str), color=C_PHII,  lw=1.4, ls='-',  alpha=0.9,
                label=r'$P_{\rm HII,Str}$', zorder=3)
    ax_top.plot(t, pos(P_HII_ODE), color=C_PHII,  lw=0.8, ls=':',  alpha=0.4,
                label=r'$P_{\rm HII,ODE}$', zorder=2)
    ax_top.plot(t, pos(P_ram),     color=C_WIND,  lw=1.2, ls='--', alpha=0.9,
                label=r'$P_{\rm ram}$', zorder=3)
    ax_top.plot(t, pos(P_rad),     color=C_RAD,   lw=1.2, ls='-.', alpha=0.9,
                label=r'$P_{\rm rad}$', zorder=3)
    ax_top.plot(t, pos(P_drive),   color='0.15',  lw=2.5, ls='-',  alpha=0.85,
                label=r'$P_{\rm drive}$', zorder=4)

    # --- Background shading by dominant driver ---
    # Where P_HII,Str > Pb: teal (HII-dominated), else: coral (bubble-dominated)
    Pb_safe = np.nan_to_num(pressures['Pb'], nan=0.0)
    Pstr_safe = np.nan_to_num(pressures['P_HII_Str'], nan=0.0)
    hii_dom = Pstr_safe > Pb_safe

    if np.any(hii_dom):
        ax_top.fill_between(t, 0, 1, where=hii_dom,
                            color=C_PHII, alpha=0.08,
                            transform=ax_top.get_xaxis_transform(), zorder=0)
    if np.any(~hii_dom):
        ax_top.fill_between(t, 0, 1, where=~hii_dom,
                            color=C_DRIVE, alpha=0.08,
                            transform=ax_top.get_xaxis_transform(), zorder=0)

    # --- Mark the P_HII,Str > Pb crossing ---
    crossings = np.where(np.diff(hii_dom.astype(int)) != 0)[0]
    for idx in crossings:
        t_cross = 0.5 * (t[idx] + t[idx + 1])
        ax_top.axvline(t_cross, color=C_PHII, lw=1.0, ls=':', alpha=0.5, zorder=1)
        # Extend crossing indicator into bottom panel
        ax_bot.axvline(t_cross, color=C_PHII, lw=1.0, ls=':', alpha=0.5, zorder=1)

    ax_top.set_yscale('log')
    ax_top.set_ylabel(r'Pressure [dyn cm$^{-2}$]')

    # Set sensible y-limits
    all_p = np.concatenate([pos(Pb), pos(P_HII_Str), pos(P_ram), pos(P_drive)])
    all_p = all_p[np.isfinite(all_p) & (all_p > 0)]
    if len(all_p) > 0:
        ax_top.set_ylim(0.3 * all_p.min(), 3.0 * all_p.max())

    ax_top.set_xlim(t.min(), t.max())

    # ================================================================
    # BOTTOM PANEL: ζ(t)
    # ================================================================
    zeta_smooth = _smooth(zeta, sw)
    zeta_pos = np.where(zeta_smooth > 0, zeta_smooth, np.nan)

    ax_bot.plot(t, zeta_pos, color='0.15', lw=1.8, ls='-', alpha=0.9,
                label=r'$\zeta(t)$', zorder=3)

    # Horizontal line at ζ = 1
    ax_bot.axhline(1.0, color='black', lw=1.5, ls='--', alpha=0.6, zorder=2)

    # Background shading matching top panel
    if np.any(hii_dom):
        ax_bot.fill_between(t, 0, 1, where=hii_dom,
                            color=C_PHII, alpha=0.08,
                            transform=ax_bot.get_xaxis_transform(), zorder=0)
    if np.any(~hii_dom):
        ax_bot.fill_between(t, 0, 1, where=~hii_dom,
                            color=C_DRIVE, alpha=0.08,
                            transform=ax_bot.get_xaxis_transform(), zorder=0)

    ax_bot.set_yscale('log')
    ax_bot.set_ylabel(r'$\zeta = R_{\rm eq}/R_{\rm St}$')
    ax_bot.set_xlabel('t [Myr]')
    ax_bot.set_xlim(t.min(), t.max())

    # Sensible y-limits for ζ
    z_valid = zeta_pos[np.isfinite(zeta_pos)]
    if len(z_valid) > 0:
        ax_bot.set_ylim(max(0.01, 0.3 * z_valid.min()),
                        min(1000, 3.0 * z_valid.max()))


# ======================================================================
# Single-run figure
# ======================================================================

def plot_from_path(data_input: str, output_dir: str = None):
    """Plot pressure + ζ figure from a single simulation path."""
    try:
        data_path = resolve_data_input(data_input, output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    try:
        t, R2, phase, pressures, zeta, rcloud, isCollapse = load_run(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 7), dpi=150,
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True,
    )

    plot_pressureZeta_on_ax(
        ax_top, ax_bot, t, R2, phase, pressures, zeta,
        rcloud=rcloud, isCollapse=isCollapse,
        smooth_window=SMOOTH_WINDOW,
        phase_change=PHASE_CHANGE,
    )
    ax_top.set_title(f"Pressure + ζ: {data_path.parent.name}")

    # Legend
    handles = _build_legend_handles()
    ax_top.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=7)

    plt.tight_layout()

    # Save
    run_name = data_path.parent.name
    parent_folder = data_path.parent.parent.name
    fig_dir = FIG_DIR / parent_folder
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = fig_dir / f"pressureZeta_{run_name}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


def _build_legend_handles():
    """Construct legend handles for the pressure + ζ plot."""
    handles = [
        Line2D([0], [0], color=C_DRIVE, lw=1.4, ls='-',
               label=r'$P_b$ (bubble)'),
        Line2D([0], [0], color=C_PHII,  lw=1.4, ls='-',
               label=r'$P_{\rm HII,Str}$ (Strömgren)'),
        Line2D([0], [0], color=C_PHII,  lw=0.8, ls=':', alpha=0.4,
               label=r'$P_{\rm HII,ODE}$ (BC)'),
        Line2D([0], [0], color=C_WIND,  lw=1.2, ls='--',
               label=r'$P_{\rm ram}$'),
        Line2D([0], [0], color=C_RAD,   lw=1.2, ls='-.',
               label=r'$P_{\rm rad}$'),
        Line2D([0], [0], color='0.15',  lw=2.5, ls='-',
               label=r'$P_{\rm drive}$'),
        Line2D([0], [0], color='0.15',  lw=1.8, ls='-',
               label=r'$\zeta(t)$'),
        Line2D([0], [0], color='black', lw=1.5, ls='--', alpha=0.6,
               label=r'$\zeta = 1$'),
    ]
    handles.extend(get_marker_legend_handles())
    return handles


# ======================================================================
# Grid figure
# ======================================================================

def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None):
    """
    Plot grid of pressure + ζ panels from simulations in a folder.

    Each grid cell contains a two-row subplot (top = pressures, bottom = ζ).
    """
    from src._output.trinity_reader import (
        find_all_simulations, organize_simulations_for_grid, get_unique_ndens
    )

    folder_path = Path(folder_path)
    folder_name = folder_path.name

    sim_files = find_all_simulations(folder_path)
    if not sim_files:
        print(f"No simulation files found in {folder_path}")
        return

    if ndens_filter:
        ndens_to_plot = [ndens_filter]
    else:
        ndens_to_plot = get_unique_ndens(sim_files)

    print(f"Found {len(sim_files)} simulations")
    print(f"  Densities to plot: {ndens_to_plot}")

    for ndens in ndens_to_plot:
        print(f"\nProcessing n={ndens}...")

        organized = organize_simulations_for_grid(
            sim_files, ndens_filter=ndens,
            mCloud_filter=mCloud_filter, sfe_filter=sfe_filter
        )
        mCloud_list = organized['mCloud_list']
        sfe_list    = organized['sfe_list']
        grid        = organized['grid']

        if not mCloud_list or not sfe_list:
            print(f"  Could not organize simulations into grid for n={ndens}")
            continue

        print(f"  mCloud: {mCloud_list}")
        print(f"  SFE: {sfe_list}")

        nrows, ncols = len(mCloud_list), len(sfe_list)

        # Outer GridSpec: one cell per (mCloud, sfe) combination
        fig = plt.figure(
            figsize=(3.5 * ncols, 3.4 * nrows),
            dpi=500,
        )
        outer_gs = GridSpec(
            nrows, ncols, figure=fig,
            hspace=0.35, wspace=0.30,
        )

        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                data_path = grid.get((mCloud, sfe))

                # Inner GridSpec: top (pressure) + bottom (ζ), ratio 3:1
                inner_gs = GridSpecFromSubplotSpec(
                    2, 1,
                    subplot_spec=outer_gs[i, j],
                    height_ratios=[3, 1],
                    hspace=0.08,
                )
                ax_top = fig.add_subplot(inner_gs[0])
                ax_bot = fig.add_subplot(inner_gs[1], sharex=ax_top)

                if data_path is None:
                    run_id = f"{mCloud}_sfe{sfe}_n{ndens}"
                    print(f"  {run_id}: missing")
                    ax_top.text(0.5, 0.5, "missing", ha="center", va="center",
                                transform=ax_top.transAxes)
                    ax_top.set_axis_off()
                    ax_bot.set_axis_off()
                    continue

                try:
                    t, R2, phase, pressures, zeta, rcloud, isCollapse = load_run(data_path)
                    plot_pressureZeta_on_ax(
                        ax_top, ax_bot, t, R2, phase, pressures, zeta,
                        rcloud=rcloud, isCollapse=isCollapse,
                        smooth_window=SMOOTH_WINDOW,
                        phase_change=PHASE_CHANGE,
                    )
                except Exception as e:
                    print(f"Error loading {data_path}: {e}")
                    ax_top.text(0.5, 0.5, "error", ha="center", va="center",
                                transform=ax_top.transAxes)
                    ax_top.set_axis_off()
                    ax_bot.set_axis_off()
                    continue

                # Hide x tick labels on top axis (shared with bottom)
                plt.setp(ax_top.get_xticklabels(), visible=False)

                # X-axis label only on bottom row
                if i < nrows - 1:
                    ax_bot.set_xlabel('')
                    plt.setp(ax_bot.get_xticklabels(), visible=False)

                # Column titles (top row only)
                if i == 0:
                    eps = int(sfe) / 100.0
                    ax_top.set_title(rf"$\epsilon={eps:.2f}$")

                # Row labels (left column only)
                if j == 0:
                    mval = float(mCloud)
                    mexp = int(np.floor(np.log10(mval)))
                    mcoeff = round(mval / (10 ** mexp))
                    if mcoeff == 10:
                        mcoeff = 1
                        mexp += 1
                    if mcoeff == 1:
                        mlabel = rf"$M_{{\rm cl}}=10^{{{mexp}}}\,M_\odot$"
                    else:
                        mlabel = rf"$M_{{\rm cl}}={mcoeff}\times10^{{{mexp}}}\,M_\odot$"
                    ax_top.set_ylabel(mlabel + "\n" + r"P [dyn cm$^{-2}$]",
                                      fontsize=7)
                else:
                    ax_top.set_ylabel('')
                    ax_bot.set_ylabel('')

        # --- Global legend ---
        handles = _build_legend_handles()
        leg = fig.legend(
            handles=handles,
            loc="upper center",
            ncol=4,
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            edgecolor="0.2",
            bbox_to_anchor=(0.5, 1.15),
            fontsize=7,
        )
        leg.set_zorder(10)

        # --- Save ---
        ndens_tag = f"n{ndens}"
        fig_dir = Path(output_dir) if output_dir else FIG_DIR / folder_name
        fig_dir.mkdir(parents=True, exist_ok=True)

        if SAVE_PDF:
            out_pdf = fig_dir / f"pressureZeta_{ndens_tag}.pdf"
            fig.savefig(out_pdf, bbox_inches="tight")
            print(f"Saved: {out_pdf}")

        plt.close(fig)


# Backwards compatibility alias
plot_folder_grid = plot_grid


# ======================================================================
# CLI dispatch
# ======================================================================

if __name__ == "__main__":
    from src._plots.cli import dispatch
    dispatch(
        script_name="paper_pressureZeta.py",
        description="Plot TRINITY pressure evolution + zeta",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
    )
