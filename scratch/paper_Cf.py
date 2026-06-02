#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Covering-fraction (Cf) trajectory plots.

Companion to paper_Rosette / paper_ODIN, for sweeps that vary the bubble
covering fraction ``coverFraction`` (Cf) while holding the initial cloud
condition fixed — e.g. the ``rosette_sweep_denser_PISM1e4_Cf`` sweep.

For every distinct initial condition (everything *except* Cf) it draws the
paper_Rosette triptych — expansion velocity v(t), bubble radius R2(t) and shell
radius rShell(t) — overlaying one curve per Cf value, coloured by Cf, with the
Rosette observational constraints drawn on each panel.  The three quantities
are the rows; each initial condition is a column, so a single-condition sweep
reproduces paper_Rosette's three stacked panels and multi-condition sweeps fan
out into a 3 x N grid.

Cf is the *closed* fraction of the bubble wall: hot gas vents through the open
area (1-Cf)*4*pi*R2^2, draining bubble energy.  Cf=1 recovers the sealed
(Weaver) bubble.  Lower Cf leaks more, so the bubble grows more slowly and may
stall/collapse earlier — exactly what these trajectories make visible.

@author: Jia Wei Teh
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent.parent))

from paper.figures._lib.plot_base import FIG_DIR              # noqa: E402
from paper.figures._lib.grid_template import filter_sim_files_by_phii  # noqa: E402
from scratch.paper_ODIN import (                              # noqa: E402
    _trim_after_end, FONTSIZE, PC_MYR_TO_KM_S, ObservationalConstraints,
)
from scratch.paper_Rosette import (                           # noqa: E402
    rosette_constraints, _V_MAX, _DUST_SHELL_MIN, _DUST_SHELL_MAX,
)
from trinity._output.trinity_reader import (                  # noqa: E402
    load_output, find_all_simulations, parse_simulation_params,
)

print("...creating covering-fraction (Cf) trajectory plots")

# Panel rows: (run-dict key, y-axis label).  Velocity is stored in km/s.
_ROWS = [
    ('v',      r'Expansion Velocity [km s$^{-1}$]'),
    ('R2',     r'Bubble Radius $R_2$ [pc]'),
    ('rShell', r'Shell Radius $r_{\rm shell}$ [pc]'),
]

# Matches the ``_coverFraction<token>`` run-name suffix produced by
# trinity._input.sweep_parser for a swept coverFraction (e.g. "0p2", "1").
_CF_TOKEN_RE = re.compile(r'_coverFraction([0-9pP.eE+\-]+)')


# =============================================================================
# Loading
# =============================================================================

def _coverFraction_of(output, folder_name: str) -> Optional[float]:
    """Covering fraction for one run.

    Prefers the run-constant in ``metadata.json`` (authoritative); falls back
    to the ``_coverFraction…`` folder-name token for older runs.  Returns
    ``None`` when neither is available.
    """
    cf = output.metadata.get('coverFraction')
    if cf is not None:
        try:
            return float(cf)
        except (TypeError, ValueError):
            pass
    m = _CF_TOKEN_RE.search(folder_name)
    if m:
        try:
            return float(m.group(1).replace('p', '.'))
        except ValueError:
            return None
    return None


def _base_condition(folder_name: str) -> str:
    """Initial-condition key: the run-folder name with the Cf token removed.

    Two runs that differ only in coverFraction strip to the same key, so they
    share a column; any other swept dimension (mCloud/sfe/nCore/profile/PHII)
    survives in the name and lands in its own column.
    """
    return _CF_TOKEN_RE.sub('', folder_name)


def load_cf_runs(folder_path: Path, phii_mode: str = 'yes',
                 trim: bool = False) -> Dict[str, List[dict]]:
    """Load Cf-sweep trajectories grouped by initial condition.

    Returns ``{base_condition: [run, ...]}`` where each run is a dict with
    ``cf`` and the three series ``v`` (km/s), ``R2`` and ``rShell``, sorted by
    ``cf`` within each group.  By default the full trajectory is kept; set
    ``trim`` True to cut each series after the first collapse/dissolution
    snapshot (drops frozen tails).  ``t_raw_max``/``n_snaps`` record the
    untrimmed span so callers can report how much trimming removed.
    """
    sim_files = find_all_simulations(folder_path)
    sim_files = filter_sim_files_by_phii(sim_files, phii_mode)
    if not sim_files:
        return {}

    groups: Dict[str, List[dict]] = {}
    for path in sim_files:
        folder_name = path.parent.name
        try:
            output = load_output(path)
        except Exception as e:
            print(f"  Error loading {path}: {e}")
            continue
        if len(output) == 0:
            continue

        cf = _coverFraction_of(output, folder_name)
        if cf is None:
            print(f"  Skipping {folder_name}: no coverFraction found")
            continue

        t = output.get('t_now')
        R2 = output.get('R2')
        if t is None or len(t) == 0 or R2 is None:
            continue
        v2 = output.get('v2')
        rShell = output.get('rShell')
        v = v2 * PC_MYR_TO_KM_S if v2 is not None else None

        t_raw = np.asarray(t, dtype=float)
        if trim:
            # Trim every series by the same collapse/dissolution cut; the
            # trimmer's M_shell slot doubles as the rShell carrier here.
            t, v, rShell, R2 = _trim_after_end(output, t, v, rShell, R2)

        groups.setdefault(_base_condition(folder_name), []).append({
            'cf': cf, 't': np.asarray(t),
            'v': np.asarray(v) if v is not None else None,
            'R2': np.asarray(R2),
            'rShell': np.asarray(rShell) if rShell is not None else None,
            'folder': folder_name,
            't_raw_max': float(t_raw.max()), 'n_snaps': int(len(t_raw)),
        })

    for runs in groups.values():
        runs.sort(key=lambda r: r['cf'])
    return groups


# =============================================================================
# Plotting
# =============================================================================

def _panel_title(base_key: str) -> str:
    """Readable column title from a base-condition key.

    Appends any folder suffix ``parse_simulation_params`` ignores (e.g. ``PL0``,
    ``yesPHII``) so columns that differ only by such a tag stay distinguishable.
    """
    p = parse_simulation_params(base_key)
    if p is None:
        return base_key
    sfe_pct = int(p['sfe']) / 100.0
    title = (rf"$M_{{\rm cloud}}={p['mCloud']}\,M_\odot$, "
             rf"sfe$={sfe_pct:g}$, $n={p['ndens']}$")
    core = f"{p['mCloud']}_sfe{p['sfe']}_n{p['ndens']}"
    idx = base_key.find(core)
    suffix = base_key[idx + len(core):].strip('_') if idx != -1 else ""
    if suffix:
        title += f" [{suffix}]"
    return title


def _sort_key(base_key: str):
    """Order columns by (mCloud, sfe, nCore) when parseable, else by name.

    ``base_key`` is the final tiebreaker so suffix-only variants (e.g.
    yesPHII/noPHII) keep a stable, distinct ordering.
    """
    p = parse_simulation_params(base_key)
    if p is None:
        return (1, base_key)
    return (0, float(p['mCloud']), int(p['sfe']), float(p['ndens']), base_key)


def _draw_obs_velocity(ax, obs: ObservationalConstraints):
    """Velocity-panel observation overlay (matches paper_Rosette)."""
    ax.errorbar(obs.t_obs, obs.v_obs, xerr=obs.t_err, yerr=obs.v_err,
                fmt='s', color='red', markersize=10, capsize=4, capthick=1.5,
                markeredgecolor='k', markeredgewidth=0.5, zorder=10,
                label=fr'$v_{{\rm exp}}$: {obs.v_obs:g}$\pm${obs.v_err:g} km/s')
    ax.axhspan(obs.v_obs - obs.v_err, obs.v_obs + obs.v_err,
               alpha=0.15, color='red', zorder=1)
    ax.axvspan(obs.t_obs - obs.t_err, obs.t_obs + obs.t_err,
               alpha=0.1, color='gray', zorder=0)


def _draw_obs_radius(ax, obs: ObservationalConstraints):
    """Radius-panel observation overlay (matches paper_Rosette)."""
    t_lo, t_hi = obs.t_obs - obs.t_err, obs.t_obs + obs.t_err
    ax.axvspan(t_lo, t_hi, alpha=0.1, color='gray', zorder=0,
               label=f'Age ({t_lo:.0f}–{t_hi:.0f} Myr)')
    ax.add_patch(Rectangle(
        (t_lo, obs.R_obs - obs.R_err), t_hi - t_lo, 2 * obs.R_err,
        facecolor='blue', alpha=0.20, edgecolor='blue', lw=1.5, zorder=5,
        label=fr'HII outer: {obs.R_obs:g}$\pm${obs.R_err:g} pc'))
    ax.add_patch(Rectangle(
        (t_lo, obs.R_obs_Pabst - obs.R_err_Pabst), t_hi - t_lo, 2 * obs.R_err_Pabst,
        facecolor='green', alpha=0.20, edgecolor='green', lw=1.5, zorder=5,
        label=fr'Cavity: {obs.R_obs_Pabst:g}$\pm${obs.R_err_Pabst:g} pc'))
    ax.add_patch(Rectangle(
        (t_lo, _DUST_SHELL_MIN), t_hi - t_lo, _DUST_SHELL_MAX - _DUST_SHELL_MIN,
        facecolor='orange', alpha=0.15, edgecolor='orange', lw=1.5, zorder=5,
        label=f'Dust shell ({_DUST_SHELL_MIN:.0f}–{_DUST_SHELL_MAX:.0f} pc)'))


def plot_cf_grid(groups: Dict[str, List[dict]], output_dir: Path,
                 folder_name: str, obs: ObservationalConstraints):
    """3 rows (v, R2, rShell) x N initial-condition columns; one line per Cf."""
    base_keys = sorted(groups, key=_sort_key)
    ncols = len(base_keys)
    nrows = len(_ROWS)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols + 1.6, 4.2 * nrows),
                             dpi=150, sharex=True, sharey='row', squeeze=False)

    # Shared Cf colour scale across all panels.
    cf_all = sorted({r['cf'] for runs in groups.values() for r in runs})
    norm = mcolors.Normalize(vmin=min(cf_all), vmax=max(cf_all))
    cmap = plt.cm.viridis

    # x-range wide enough to show both the trajectories and the (later-time)
    # observation markers, which sit at t up to t_obs + t_err.
    t_data_max = max((r['t'].max() for runs in groups.values() for r in runs),
                     default=1.0)
    x_hi = max(t_data_max, obs.t_obs + obs.t_err) * 1.05

    for col, base_key in enumerate(base_keys):
        for row, (key, _ylabel) in enumerate(_ROWS):
            ax = axes[row, col]
            for run in groups[base_key]:
                y = run[key]
                if y is not None:
                    ax.plot(run['t'], y, color=cmap(norm(run['cf'])),
                            lw=1.6, alpha=0.9)
            # Observation overlay per panel.
            _draw_obs_velocity(ax, obs) if key == 'v' else _draw_obs_radius(ax, obs)
            ax.grid(False)
            ax.tick_params(axis='both', labelsize=FONTSIZE - 4)
        axes[0, col].set_title(_panel_title(base_key), fontsize=FONTSIZE - 3)

    # Set limits only after every curve + obs patch is drawn: an explicit
    # set_ylim disables autoscale for the whole shared row, so doing it mid-loop
    # would freeze the y-range to the first column and clip taller later columns.
    for col in range(ncols):
        for row in range(nrows):
            axes[row, col].set_xlim(0, x_hi)
        axes[0, col].set_ylim(0, _V_MAX)          # velocity capped like paper_Rosette
        axes[1, col].set_ylim(bottom=0)
        axes[2, col].set_ylim(bottom=0)

    # Row labels (left column) and time labels (bottom row).
    for row, (_key, ylabel) in enumerate(_ROWS):
        axes[row, 0].set_ylabel(ylabel, fontsize=FONTSIZE)
    for col in range(ncols):
        axes[-1, col].set_xlabel('Time [Myr]', fontsize=FONTSIZE)

    # Observation legends, once per row on the left column.
    axes[0, 0].legend(loc='upper right', fontsize=FONTSIZE - 6)
    axes[1, 0].legend(loc='upper left', fontsize=FONTSIZE - 6)
    axes[2, 0].legend(loc='upper left', fontsize=FONTSIZE - 6)

    # Shared Cf colourbar, ticked at the actual Cf values.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02,
                        ticks=cf_all)
    cbar.set_label(r'Covering fraction $C_f$', fontsize=FONTSIZE)
    cbar.ax.tick_params(labelsize=FONTSIZE - 4)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = output_dir / f'cf_trajectory_{folder_name}.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


# =============================================================================
# Main / CLI
# =============================================================================

def main(folder_path: str, output_dir: Optional[str] = None,
         phii_mode: str = 'yes', trim: bool = False):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    folder_name = folder_path.name
    out_dir = Path(output_dir) if output_dir else FIG_DIR / folder_name

    print(f"\nLoading simulations from: {folder_path}")
    groups = load_cf_runs(folder_path, phii_mode, trim)
    if not groups:
        print("No valid Cf simulation results found.")
        sys.exit(1)

    n_runs = sum(len(v) for v in groups.values())
    print(f"Loaded {n_runs} runs across {len(groups)} initial condition(s)"
          f"  (trim={'on' if trim else 'off'})")

    # Per-column diagnostic: how many Cf values landed in each condition, and
    # how far the plotted series reaches vs the full (untrimmed) run. A big gap
    # between t_plotted and t_full means the collapse/dissolution trimmer is
    # what's shrinking the x-axis — run without --trim to see the full curve.
    for base_key in sorted(groups, key=_sort_key):
        runs = groups[base_key]
        cfs = ", ".join(f"{r['cf']:g}" for r in runs)
        t_plot = max(r['t'].max() for r in runs)
        t_raw = max(r['t_raw_max'] for r in runs)
        print(f"  {base_key}: {len(runs)} run(s), Cf=[{cfs}], "
              f"t_plotted_max={t_plot:.4g} Myr, t_full_max={t_raw:.4g} Myr")
    print(f"Output directory: {out_dir}")

    plot_cf_grid(groups, out_dir, folder_name, rosette_constraints())
    print("\nDone!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TRINITY covering-fraction (Cf) trajectory plots "
                    "(velocity, bubble radius, shell radius vs Rosette obs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_Cf.py --folder outputs/rosette_sweep_denser_PISM1e4_Cf
  python paper_Cf.py --folder outputs/rosette_sweep_denser_PISM1e4_Cf --trim
""")
    parser.add_argument('--folder', '-F', required=True,
                        help='Path to sweep output folder')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output directory (default: fig/{folder})')
    parser.add_argument('--phii', choices=['yes', 'no'], default='yes',
                        help="PHII folder filter: 'yes' keeps yesPHII+untagged, "
                             "'no' keeps only noPHII (default: yes)")
    parser.add_argument('--trim', dest='trim', action='store_true',
                        help='Cut each trajectory at the first collapse/dissolution '
                             'snapshot (default: keep the full trajectory)')

    args = parser.parse_args()
    main(args.folder, args.output_dir, args.phii, args.trim)
