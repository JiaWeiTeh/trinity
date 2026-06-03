#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Covering-fraction (Cf) trajectory and constraint plots.

Companion to paper_Rosette / paper_ODIN, for sweeps that vary the bubble
covering fraction ``coverFraction`` (Cf) while holding the initial cloud
condition fixed — e.g. the ``rosette_sweep_denser_PISM1e4_Cf`` sweep.

Produces two figures per run, both grouped by initial condition (everything
*except* Cf), with one column per condition (so a single-condition sweep
reproduces paper_Rosette's three stacked panels and multi-condition sweeps fan
out into a 3 x N grid):

1. ``cf_trajectory_*`` — the paper_Rosette triptych v(t) / R2(t) / rShell(t),
   one curve per Cf coloured by Cf.  The cluster age is assumed *fixed*, so the
   observations are drawn as y-only error bars AT that age (with a thin age
   uncertainty span), not as time-spanning boxes — a box would let a curve
   "match" by crossing it at the wrong epoch.

2. ``cf_constraint_*`` — the observable evaluated at the assumed age vs Cf, with
   the measured value as a horizontal band.  Where the model curve enters the
   band tells you which Cf reproduces the observation.  The grey band around the
   model curve is its spread over the age uncertainty.

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

print("...creating covering-fraction (Cf) plots")

# Panel rows: (run-dict key, y-axis label).  Velocity is stored in km/s.
_ROWS = [
    ('v',      r'Expansion Velocity [km s$^{-1}$]'),
    ('R2',     r'Bubble Radius $R_2$ [pc]'),
    ('rShell', r'Shell Radius $r_{\rm shell}$ [pc]'),
]

# Both radius panels share a fixed 0-25 pc window so columns are comparable and
# the (<=22 pc) observation bands stay in view.
_RADIUS_YLIM = (0.0, 25.0)

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
# Labels / ordering
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


def _cf_ticks(cf_all, step: float = 0.05):
    """Colourbar ticks at multiples of ``step`` within the swept Cf range."""
    lo, hi = min(cf_all), max(cf_all)
    start = np.ceil(lo / step - 1e-9) * step
    ticks = np.round(np.arange(start, hi + 1e-9, step), 10)
    return ticks if len(ticks) else np.array([lo])


# =============================================================================
# Observation overlays
# =============================================================================

def _draw_age_marker(ax, obs: ObservationalConstraints):
    """Vertical line at the assumed age, with a thin age-uncertainty span."""
    if obs.t_err > 0:
        ax.axvspan(obs.t_obs - obs.t_err, obs.t_obs + obs.t_err,
                   color='gray', alpha=0.12, zorder=0)
    ax.axvline(obs.t_obs, color='k', ls='--', lw=1.2, alpha=0.7, zorder=2,
               label=f'Assumed age {obs.t_obs:g} Myr')

# =============================================================================
# Observation overlays
# =============================================================================

def _obs_points(key: str, obs: ObservationalConstraints):
    """(value, error, colour, marker, label) tuples for the observation(s)
    that constrain a given row.

    Each radius panel shows only its own observable.  Per TRINITY's variable
    definitions — R2 is the outer bubble radius (= inner shell edge), rShell is
    the (outer) shell radius — R2 is compared to the cavity inner radius, and
    rShell to the HII outer radius and dust shell.
    """
    if key == 'v':
        return [(obs.v_obs, obs.v_err, 'red', 's',
                 fr'$v_{{\rm exp}}$: {obs.v_obs:g}$\pm${obs.v_err:g} km/s')]
    if key == 'R2':
        return [(obs.R_obs_Pabst, obs.R_err_Pabst, 'green', '^',
                 fr'Cavity: {obs.R_obs_Pabst:g}$\pm${obs.R_err_Pabst:g} pc')]
    # rShell
    dust_mid = 0.5 * (_DUST_SHELL_MIN + _DUST_SHELL_MAX)
    dust_err = 0.5 * (_DUST_SHELL_MAX - _DUST_SHELL_MIN)
    return [
        (obs.R_obs, obs.R_err, 'blue', 'o',
         fr'HII outer: {obs.R_obs:g}$\pm${obs.R_err:g} pc'),
        (dust_mid, dust_err, 'orange', 'D',
         f'Dust shell ({_DUST_SHELL_MIN:.0f}–{_DUST_SHELL_MAX:.0f} pc)'),
    ]


def _draw_obs_at_age(ax, key: str, obs: ObservationalConstraints):
    """Trajectory-panel overlay: error bar(s) at the assumed age (y-only)."""
    _draw_age_marker(ax, obs)
    for val, err, color, marker, label in _obs_points(key, obs):
        ax.errorbar(obs.t_obs, val, yerr=err, fmt=marker, color=color,
                    markersize=9, capsize=4, capthick=1.5,
                    markeredgecolor='k', markeredgewidth=0.5, zorder=10,
                    label=label)


def _draw_obs_hbands(ax, key: str, obs: ObservationalConstraints):
    """Constraint-panel overlay: measured value(s) as horizontal band(s)."""
    for val, err, color, _marker, label in _obs_points(key, obs):
        ax.axhspan(val - err, val + err, color=color, alpha=0.18, zorder=0,
                   label=label)
        ax.axhline(val, color=color, lw=1.0, ls='--', alpha=0.6, zorder=1)


# =============================================================================
# Trajectory plot
# =============================================================================

def plot_cf_trajectory(groups: Dict[str, List[dict]], output_dir: Path,
                       folder_name: str, obs: ObservationalConstraints):
    """3 rows (v, R2, rShell) x N initial-condition columns; one line per Cf,
    with observations drawn at the assumed age."""
    base_keys = sorted(groups, key=_sort_key)
    ncols = len(base_keys)
    nrows = len(_ROWS)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols + 1.6, 4.2 * nrows),
                             dpi=150, sharex=True, sharey='row', squeeze=False)

    cf_all = sorted({r['cf'] for runs in groups.values() for r in runs})
    norm = mcolors.Normalize(vmin=min(cf_all), vmax=max(cf_all))
    cmap = plt.cm.viridis

    t_data_max = max((r['t'].max() for runs in groups.values() for r in runs),
                     default=1.0)
    x_hi = max(t_data_max, obs.t_obs + obs.t_err) * 1.02

    for col, base_key in enumerate(base_keys):
        for row, (key, _ylabel) in enumerate(_ROWS):
            ax = axes[row, col]
            for run in groups[base_key]:
                y = run[key]
                if y is not None:
                    ax.plot(run['t'], y, color=cmap(norm(run['cf'])),
                            lw=1.6, alpha=0.9)
            _draw_obs_at_age(ax, key, obs)
            ax.grid(False)
            ax.tick_params(axis='both', labelsize=FONTSIZE - 4)
        axes[0, col].set_title(_panel_title(base_key), fontsize=FONTSIZE - 3)

    # Limits after all curves+markers are drawn (a mid-loop set_ylim would
    # freeze the shared-row y-range to the first column and clip taller ones).
    for col in range(ncols):
        for row in range(nrows):
            axes[row, col].set_xlim(0, x_hi)
        axes[0, col].set_ylim(0, _V_MAX)          # velocity capped like paper_Rosette
        axes[1, col].set_ylim(*_RADIUS_YLIM)      # bubble radius (0-25 pc)
        axes[2, col].set_ylim(*_RADIUS_YLIM)      # shell radius  (0-25 pc)

    for row, (_key, ylabel) in enumerate(_ROWS):
        axes[row, 0].set_ylabel(ylabel, fontsize=FONTSIZE)
    for col in range(ncols):
        axes[-1, col].set_xlabel('Time [Myr]', fontsize=FONTSIZE)
    for row in range(nrows):
        axes[row, 0].legend(loc='upper left', fontsize=FONTSIZE - 6)

    # Colourbar spanning the full height of all three rows, ticked at 0.05.
    fig.tight_layout(rect=[0, 0, 0.90, 1.0])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    pos_top = axes[0, -1].get_position()
    pos_bot = axes[-1, -1].get_position()
    cax = fig.add_axes([0.915, pos_bot.y0, 0.022, pos_top.y1 - pos_bot.y0])
    cbar = fig.colorbar(sm, cax=cax, ticks=_cf_ticks(cf_all))
    cbar.set_label(r'Covering fraction $C_f$', fontsize=FONTSIZE)
    cbar.ax.tick_params(labelsize=FONTSIZE - 4)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = output_dir / f'cf_trajectory_{folder_name}.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


# =============================================================================
# Constraint plot (observable at the assumed age vs Cf)
# =============================================================================

def _value_at_age(run: dict, key: str, age: float, age_err: float):
    """Return (value_at_age, lo, hi) for one observable, where [lo, hi] is the
    model's spread over the age window.  Returns NaNs when the run does not
    cover the age (so clamped/extrapolated values are not plotted as matches)."""
    y = run[key]
    t = run['t']
    if y is None or len(t) < 2 or age < t.min() or age > t.max():
        return np.nan, np.nan, np.nan
    val = float(np.interp(age, t, y))
    grid = np.linspace(max(t.min(), age - age_err), min(t.max(), age + age_err), 9)
    w = np.interp(grid, t, y)
    return val, float(w.min()), float(w.max())


def plot_cf_constraint(groups: Dict[str, List[dict]], output_dir: Path,
                       folder_name: str, obs: ObservationalConstraints):
    """3 rows (v, R2, rShell) x N columns of observable-at-age vs Cf, with the
    measured value as a horizontal band.  The model/obs band overlap gives the
    Cf range consistent with each observation."""
    base_keys = sorted(groups, key=_sort_key)
    ncols = len(base_keys)
    nrows = len(_ROWS)
    age, age_err = obs.t_obs, obs.t_err

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols + 0.4, 4.2 * nrows),
                             dpi=150, sharex=True, sharey='row', squeeze=False)

    ylabels = [
        fr'Velocity at {age:g} Myr [km s$^{{-1}}$]',
        fr'$R_2$ at {age:g} Myr [pc]',
        fr'$r_{{\rm shell}}$ at {age:g} Myr [pc]',
    ]

    for col, base_key in enumerate(base_keys):
        runs = groups[base_key]
        cfs = np.array([r['cf'] for r in runs])
        for row, (key, _ylabel) in enumerate(_ROWS):
            ax = axes[row, col]
            vals, los, his = zip(*(_value_at_age(r, key, age, age_err) for r in runs))
            vals, los, his = np.array(vals), np.array(los), np.array(his)
            if age_err > 0 and np.any(np.isfinite(los)):
                ax.fill_between(cfs, los, his, color='0.6', alpha=0.25, zorder=1,
                                label=f'age $\\pm$ {age_err:g} Myr')
            ax.plot(cfs, vals, '-', color='0.5', lw=1.0, zorder=2)
            ax.plot(cfs, vals, 'o', color='C0', markersize=6,
                    markeredgecolor='k', markeredgewidth=0.4, zorder=3)
            _draw_obs_hbands(ax, key, obs)
            ax.set_ylim(bottom=0)
            ax.grid(False)
            ax.tick_params(axis='both', labelsize=FONTSIZE - 4)
        axes[0, col].set_title(_panel_title(base_key), fontsize=FONTSIZE - 3)

    for row in range(nrows):
        axes[row, 0].set_ylabel(ylabels[row], fontsize=FONTSIZE)
        axes[row, 0].legend(loc='upper left', fontsize=FONTSIZE - 6)
    for col in range(ncols):
        axes[-1, col].set_xlabel(r'Covering fraction $C_f$', fontsize=FONTSIZE)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = output_dir / f'cf_constraint_{folder_name}.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


# =============================================================================
# Main / CLI
# =============================================================================

def main(folder_path: str, output_dir: Optional[str] = None,
         phii_mode: str = 'yes', trim: bool = False,
         age: float = 2.0, age_err: float = 0.5):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    folder_name = folder_path.name
    out_dir = Path(output_dir) if output_dir else FIG_DIR / folder_name

    # Rosette constraints, but with the cluster age fixed to the assumed value.
    obs = rosette_constraints()
    obs.t_obs, obs.t_err = age, age_err

    print(f"\nLoading simulations from: {folder_path}")
    groups = load_cf_runs(folder_path, phii_mode, trim)
    if not groups:
        print("No valid Cf simulation results found.")
        sys.exit(1)

    n_runs = sum(len(v) for v in groups.values())
    print(f"Loaded {n_runs} runs across {len(groups)} initial condition(s)"
          f"  (trim={'on' if trim else 'off'}, age={age:g}±{age_err:g} Myr)")

    # Per-column diagnostic: Cf membership, plotted vs full time span, and
    # whether each condition's runs actually reach the assumed age.
    for base_key in sorted(groups, key=_sort_key):
        runs = groups[base_key]
        cfs = ", ".join(f"{r['cf']:g}" for r in runs)
        t_plot = max(r['t'].max() for r in runs)
        t_raw = max(r['t_raw_max'] for r in runs)
        n_reach = sum(r['t'].max() >= age for r in runs)
        print(f"  {base_key}: {len(runs)} run(s), Cf=[{cfs}], "
              f"t_plotted_max={t_plot:.4g} Myr, t_full_max={t_raw:.4g} Myr, "
              f"{n_reach}/{len(runs)} reach {age:g} Myr")
    print(f"Output directory: {out_dir}")

    plot_cf_trajectory(groups, out_dir, folder_name, obs)
    plot_cf_constraint(groups, out_dir, folder_name, obs)
    print("\nDone!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TRINITY covering-fraction (Cf) plots: trajectory triptych "
                    "+ observable-vs-Cf constraint at a fixed cluster age",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_Cf.py --folder outputs/rosette_sweep_denser_PISM1e4_Cf
  python paper_Cf.py --folder outputs/rosette_sweep_denser_PISM1e4_Cf --age 4 --age-err 2
""")
    parser.add_argument('--folder', '-F', required=True,
                        help='Path to sweep output folder')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output directory (default: fig/{folder})')
    parser.add_argument('--age', type=float, default=2.0,
                        help='Assumed cluster age [Myr] for the observation '
                             'comparison (default: 2.0)')
    parser.add_argument('--age-err', type=float, default=0.5,
                        help='Age uncertainty [Myr] (default: 0.5)')
    parser.add_argument('--phii', choices=['yes', 'no'], default='yes',
                        help="PHII folder filter: 'yes' keeps yesPHII+untagged, "
                             "'no' keeps only noPHII (default: yes)")
    parser.add_argument('--trim', dest='trim', action='store_true',
                        help='Cut each trajectory at the first collapse/dissolution '
                             'snapshot (default: keep the full trajectory)')

    args = parser.parse_args()
    main(args.folder, args.output_dir, args.phii, args.trim, args.age, args.age_err)
