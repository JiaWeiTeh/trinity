#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Covering-fraction (Cf) radius-trajectory plots.

Companion to paper_Rosette / paper_ODIN, for sweeps that vary the bubble
covering fraction ``coverFraction`` (Cf) while holding the initial cloud
condition fixed — e.g. the ``rosette_sweep_denser_PISM1e4_Cf`` sweep.

For every distinct initial condition (everything *except* Cf) it overlays the
radius trajectory R(t) of each Cf value on a single axes, coloured by Cf.
When the sweep contains more than one initial condition, each condition gets
its own panel on a shared grid.

Cf is the *closed* fraction of the bubble wall: hot gas vents through the open
area (1-Cf)*4*pi*R2^2, draining bubble energy.  Cf=1 recovers the sealed
(Weaver) bubble.  Lower Cf leaks more, so the bubble grows more slowly and may
stall/collapse earlier — exactly what these trajectories make visible.

@author: Jia Wei Teh
"""

import re
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

sys.path.insert(0, str(Path(__file__).parent.parent))

from paper.figures._lib.plot_base import FIG_DIR              # noqa: E402
from paper.figures._lib.grid_template import filter_sim_files_by_phii  # noqa: E402
from scratch.paper_ODIN import _trim_after_end, FONTSIZE      # noqa: E402
from trinity._output.trinity_reader import (                  # noqa: E402
    load_output, find_all_simulations, parse_simulation_params,
)

print("...creating covering-fraction (Cf) radius-trajectory plots")

# Radius quantities the reader exposes, with display labels.
_RADIUS_LABELS = {
    'R2': r'Bubble radius $R_2$ [pc]',
    'rShell': r'Shell radius $r_{\rm shell}$ [pc]',
}

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
    share a panel; any other swept dimension (mCloud/sfe/nCore/profile/PHII)
    survives in the name and lands in its own panel.
    """
    return _CF_TOKEN_RE.sub('', folder_name)


def load_cf_runs(folder_path: Path, radius_key: str, phii_mode: str = 'yes',
                 trim: bool = False) -> Dict[str, List[dict]]:
    """Load Cf-sweep trajectories grouped by initial condition.

    Returns ``{base_condition: [run, ...]}`` where each run is a dict with
    ``cf``, ``t`` and ``R`` (the chosen radius series), sorted by ``cf`` within
    each group.  By default the full trajectory is kept; set ``trim`` True to
    cut each series after the first collapse/dissolution snapshot (drops frozen
    tails).  ``t_raw_max``/``n_snaps`` record the untrimmed span so callers can
    report how much trimming removed.
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
        R = output.get(radius_key)
        if R is None or t is None or len(t) == 0:
            continue
        t_raw = np.asarray(t, dtype=float)
        if trim:
            # Reuse the ODIN trimmer to drop frozen tails after collapse/dissolve.
            t, _, _, R = _trim_after_end(output, t, None, None, R)
        if R is None:
            continue

        groups.setdefault(_base_condition(folder_name), []).append(
            {'cf': cf, 't': np.asarray(t), 'R': np.asarray(R), 'folder': folder_name,
             't_raw_max': float(t_raw.max()), 'n_snaps': int(len(t_raw))})

    for runs in groups.values():
        runs.sort(key=lambda r: r['cf'])
    return groups


# =============================================================================
# Plotting
# =============================================================================

def _panel_title(base_key: str) -> str:
    """Readable panel title from a base-condition key."""
    p = parse_simulation_params(base_key)
    if p is None:
        return base_key
    sfe_pct = int(p['sfe']) / 100.0
    return (rf"$M_{{\rm cloud}}={p['mCloud']}\,M_\odot$, "
            rf"sfe$={sfe_pct:g}$, $n={p['ndens']}$")


def _sort_key(base_key: str):
    """Order panels by (mCloud, sfe, nCore) when parseable, else by name."""
    p = parse_simulation_params(base_key)
    if p is None:
        return (1, base_key)
    return (0, float(p['mCloud']), int(p['sfe']), float(p['ndens']))


def plot_cf_grid(groups: Dict[str, List[dict]], output_dir: Path,
                 folder_name: str, radius_key: str):
    """One panel per initial condition; one R(t) line per Cf, coloured by Cf."""
    base_keys = sorted(groups, key=_sort_key)
    n = len(base_keys)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols + 1.5, 4.0 * nrows),
                             dpi=150, sharex=True, sharey=True, squeeze=False)
    flat_axes = axes.ravel()

    # Shared Cf colour scale across all panels.
    cf_all = sorted({r['cf'] for runs in groups.values() for r in runs})
    norm = mcolors.Normalize(vmin=min(cf_all), vmax=max(cf_all))
    cmap = plt.cm.viridis

    for ax, base_key in zip(flat_axes, base_keys):
        for run in groups[base_key]:
            ax.plot(run['t'], run['R'], color=cmap(norm(run['cf'])),
                    lw=1.8, alpha=0.9)
        ax.set_title(_panel_title(base_key), fontsize=FONTSIZE - 3)
        ax.tick_params(axis='both', labelsize=FONTSIZE - 4)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(False)

    # Hide unused grid cells.
    for ax in flat_axes[n:]:
        ax.set_visible(False)

    # Shared axis labels along the left column / bottom row.
    for ax in axes[-1, :]:
        ax.set_xlabel('Time [Myr]', fontsize=FONTSIZE)
    for ax in axes[:, 0]:
        ax.set_ylabel(_RADIUS_LABELS[radius_key], fontsize=FONTSIZE)

    # Shared colourbar, ticked at the actual Cf values.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                        ticks=cf_all)
    cbar.set_label(r'Covering fraction $C_f$', fontsize=FONTSIZE)
    cbar.ax.tick_params(labelsize=FONTSIZE - 4)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = output_dir / f'cf_{radius_key}_trajectory_{folder_name}.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


# =============================================================================
# Main / CLI
# =============================================================================

def main(folder_path: str, output_dir: Optional[str] = None,
         radius_key: str = 'R2', phii_mode: str = 'yes', trim: bool = False):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    folder_name = folder_path.name
    out_dir = Path(output_dir) if output_dir else FIG_DIR / folder_name

    print(f"\nLoading simulations from: {folder_path}")
    groups = load_cf_runs(folder_path, radius_key, phii_mode, trim)
    if not groups:
        print("No valid Cf simulation results found.")
        sys.exit(1)

    n_runs = sum(len(v) for v in groups.values())
    print(f"Loaded {n_runs} runs across {len(groups)} initial condition(s)"
          f"  (trim={'on' if trim else 'off'})")

    # Per-panel diagnostic: how many Cf values landed in each condition, and
    # how far the plotted series reaches vs the full (untrimmed) run. A big gap
    # between t_plotted and t_raw means the collapse/dissolution trimmer is
    # what's shrinking the x-axis — rerun with --no-trim to see the full curve.
    for base_key in sorted(groups, key=_sort_key):
        runs = groups[base_key]
        cfs = ", ".join(f"{r['cf']:g}" for r in runs)
        t_plot = max(r['t'].max() for r in runs)
        t_raw = max(r['t_raw_max'] for r in runs)
        print(f"  {base_key}: {len(runs)} run(s), Cf=[{cfs}], "
              f"t_plotted_max={t_plot:.4g} Myr, t_full_max={t_raw:.4g} Myr")
    print(f"Output directory: {out_dir}")

    plot_cf_grid(groups, out_dir, folder_name, radius_key)
    print("\nDone!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TRINITY covering-fraction (Cf) radius-trajectory plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_Cf.py --folder outputs/rosette_sweep_denser_PISM1e4_Cf
  python paper_Cf.py --folder outputs/rosette_sweep_denser_PISM1e4_Cf --radius rShell
""")
    parser.add_argument('--folder', '-F', required=True,
                        help='Path to sweep output folder')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output directory (default: fig/{folder})')
    parser.add_argument('--radius', choices=sorted(_RADIUS_LABELS), default='R2',
                        help='Radius series to plot (default: R2)')
    parser.add_argument('--phii', choices=['yes', 'no'], default='yes',
                        help="PHII folder filter: 'yes' keeps yesPHII+untagged, "
                             "'no' keeps only noPHII (default: yes)")
    parser.add_argument('--trim', dest='trim', action='store_true',
                        help='Cut each trajectory at the first collapse/dissolution '
                             'snapshot (default: keep the full trajectory)')

    args = parser.parse_args()
    main(args.folder, args.output_dir, args.radius, args.phii, args.trim)
