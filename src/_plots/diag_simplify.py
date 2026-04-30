#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic plot: pre/post simplification for the profile arrays in
``outputs/mockOutput``.

Lays out a 3×4 grid showing the three heaviest bubble arrays plus the
shell density profile across the three pre-trim phases
(``1_begin``, ``2_energy``, ``3_implicit``):

  Cols:  log_bubble_T_arr | log_bubble_n_arr | log_bubble_dTdr_arr | log_shell_n_arr
  Rows:  1_begin / 2_energy / 3_implicit

Each panel overlays:

    * Original — the saved (large-N) array drawn as a thick, low-alpha
      grey line.
    * Simplified — re-run through ``_simplify`` and drawn as filled
      markers connected by a thin red line.

Reconstruction R² and compression ratio are annotated in each panel.

Output goes to ``fig/diag/`` (created on first use).  The directory
mirrors the project-wide ``FIG_DIR`` convention from ``plot_base``
without importing it (``plot_base`` loads the LaTeX-based paper
mplstyle, which is overkill for a diagnostic plot).

Usage
-----
    python -m src._plots.diag_simplify
    python -m src._plots.diag_simplify --run path/to/mockOutput/<run-name>
    python -m src._plots.diag_simplify --npoints 200
    python -m src._plots.diag_simplify --output custom.png --no-show

The default mock-run path is
``outputs/mockOutput/1e6_sfe001_n1e3_PL0_yesPHII``; without ``--output``
the figure is saved to ``fig/diag/diag_simplify_<run-name>.png``
(or ``..._n<N>.png`` when ``--npoints`` differs from the default).

@author: TRINITY Team
"""

from __future__ import annotations

import argparse
import json
import sys as _sys
import warnings
from pathlib import Path

# Repo-root sys.path shim, matching ``diagnostic_parameter_changes.py``.
# We deliberately avoid ``plot_base`` here: that module loads the
# trinity paper mplstyle (which requires LaTeX) — overkill for a
# diagnostic plot.  We mirror its ``FIG_DIR`` definition instead.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src._functions.simplify import _simplify, _simplify_error

# Canonical output directory for diagnostic plots
# (``<project_root>/fig/diag/``).  Mirrors ``FIG_DIR`` from
# ``src._plots.plot_base`` but with a ``diag/`` sub-folder.
FIG_DIR = _PROJECT_ROOT / "fig"
DIAG_DIR = FIG_DIR / "diag"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


# Arrays to plot per snapshot. Each entry is (y_key, x_key, axis_label).
# Three bubble columns + one shell column.  Shell arrays in the saved
# files are already small (≤ 100 pts) because the *previous* simplify
# implementation reduced them aggressively; with the current code they
# would re-simplify to ``min(N_in, 100)``, so the panel typically shows
# a pass-through.  See the figure caption for context.
_ARRAY_TRIPLETS = [
    ("log_bubble_T_arr",    "bubble_T_arr_r_arr",    r"$\log_{10}\,T_{\mathrm{bub}}$"),
    ("log_bubble_n_arr",    "bubble_n_arr_r_arr",    r"$\log_{10}\,n_{\mathrm{bub}}$"),
    ("log_bubble_dTdr_arr", "bubble_dTdr_arr_r_arr", r"$\log_{10}|\,dT/dr|$"),
    ("log_shell_n_arr",     "shell_r_arr",           r"$\log_{10}\,n_{\mathrm{sh}}$"),
]

# Phase snapshots laid out top-to-bottom.
_PHASES = ("1_begin", "2_energy", "3_implicit")

_DEFAULT_RUN = "outputs/mockOutput/1e6_sfe001_n1e3_PL0_yesPHII"
_DEFAULT_NPOINTS = 100


def _load_snapshot(run_dir: Path, phase: str) -> dict:
    """Read one snapshot from the run directory."""
    path = run_dir / f"{phase}.jsonl"
    with open(path) as f:
        return json.loads(f.readline())


def _plot_panel(ax, x_orig, y_orig, x_simp, y_simp, *, ylabel: str, title: str):
    """Render one (pre, post) panel into ``ax``."""
    ax.plot(x_orig, y_orig, color="0.55", lw=2.4, alpha=0.45,
            label=f"original ({len(x_orig)} pts)", zorder=1)
    ax.plot(x_simp, y_simp, "o-", color="tab:red", ms=3.5, lw=0.9,
            markeredgecolor="darkred", markeredgewidth=0.4,
            label=f"simplified ({len(x_simp)} pts)", zorder=3)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"$r$ [pc]", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(False)
    ax.legend(loc="best", fontsize=7, framealpha=0.85)

    # Annotate R² + compression in a corner box.
    metrics = _simplify_error(x_orig, y_orig, x_simp, y_simp)
    txt = (
        rf"$R^2 = {metrics['r_squared']:.4f}$"
        + "\n"
        + rf"compression $= {metrics['compression']:.1f}\times$"
    )
    ax.text(
        0.97, 0.05, txt,
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="0.7",
                  alpha=0.85),
    )


def make_diag_grid(run_dir: Path, output: Path | None = None,
                   show: bool = True,
                   npoints: int = _DEFAULT_NPOINTS) -> None:
    """Build the 3×4 diagnostic grid for the given mock-run directory."""
    n_cols = len(_ARRAY_TRIPLETS)
    fig, axes = plt.subplots(
        3, n_cols, figsize=(3.6 * n_cols, 10),
        constrained_layout=True,
    )

    for row, phase in enumerate(_PHASES):
        try:
            snap = _load_snapshot(run_dir, phase)
        except FileNotFoundError:
            for col in range(n_cols):
                axes[row, col].set_visible(False)
            continue

        for col, (ykey, xkey, ylabel) in enumerate(_ARRAY_TRIPLETS):
            ax = axes[row, col]
            if xkey not in snap or ykey not in snap:
                ax.set_visible(False)
                continue

            x_raw = np.asarray(snap[xkey], dtype=float)
            y_raw = np.asarray(snap[ykey], dtype=float)

            if x_raw.size == 0:
                ax.set_visible(False)
                continue

            # Sort ascending for plotting (bubble grids are stored descending).
            order = np.argsort(x_raw)
            x_orig = x_raw[order]
            y_orig = y_raw[order]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                x_simp, y_simp = _simplify(x_orig, y_orig, nmin=npoints)

            short_y = (ykey.replace("_arr", "")
                       .replace("log_bubble_", "")
                       .replace("log_shell_", "shell_"))
            title = f"{phase} — {short_y}"
            _plot_panel(ax, x_orig, y_orig, x_simp, y_simp,
                        ylabel=ylabel, title=title)

    fig.suptitle(
        f"_simplify diagnostic: pre vs post on {run_dir.name}\n"
        f"(saved shell arrays are already from the OLD simplifier — "
        f"the current code passes them through unchanged)",
        fontsize=11,
    )

    if output is not None:
        fig.savefig(output, dpi=140)
        print(f"Saved figure to {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Render a 3×4 grid comparing original and simplified "
                     "bubble profile arrays from a mockOutput run."),
    )
    parser.add_argument(
        "--run", type=Path, default=Path(_DEFAULT_RUN),
        help=f"Mock-run directory (default: {_DEFAULT_RUN})",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Override the output path.  Default: "
             "fig/diag/diag_simplify_<run_name>.png",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not display the figure interactively (useful for batch).",
    )
    parser.add_argument(
        "--npoints", "-n", type=int, default=_DEFAULT_NPOINTS,
        help=(f"Target number of simplified points per panel "
              f"(default: {_DEFAULT_NPOINTS}). _simplify enforces a "
              f"floor of 20 (matches the coverage-skeleton chunk "
              f"count); values below 20 are clamped."),
    )
    args = parser.parse_args(argv)

    run_dir = args.run.resolve()
    if not run_dir.exists():
        parser.error(f"run directory not found: {run_dir}")

    output = args.output
    if output is None:
        suffix = (f"_n{args.npoints}"
                  if args.npoints != _DEFAULT_NPOINTS else "")
        output = DIAG_DIR / f"diag_simplify_{run_dir.name}{suffix}.png"

    make_diag_grid(run_dir, output=output, show=not args.no_show,
                   npoints=args.npoints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
