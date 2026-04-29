#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic plot: pre/post simplification for the bubble profile arrays
in ``outputs/mockOutput``.

Lays out a 3×3 grid showing the three heaviest bubble arrays
(``log_bubble_T_arr``, ``log_bubble_n_arr``, ``log_bubble_dTdr_arr``)
across the three pre-trim phases (``1_begin``, ``2_energy``,
``3_implicit``).  Each panel overlays:

    * Original — the saved (large-N) array drawn as a thick, low-alpha
      grey line.
    * Simplified — re-run through ``_simplify`` and drawn as filled
      markers connected by a thin red line.

Reconstruction R² and compression ratio are annotated in each panel.

Usage
-----
    python -m src._plots.diag_simplify
    python -m src._plots.diag_simplify --output diag_simplify.png
    python -m src._plots.diag_simplify --run path/to/mockOutput/<run-name>

The default mock-run path is
``outputs/mockOutput/1e6_sfe001_n1e3_PL0_yesPHII``.

@author: TRINITY Team
"""

from __future__ import annotations

import argparse
import json
import sys as _sys
import warnings
from pathlib import Path

# Repo-root sys.path shim, matching other src._plots scripts.
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from src._functions.simplify import _simplify, _simplify_error


# Arrays to plot per snapshot. Each entry is (y_key, x_key, axis_label).
_ARRAY_TRIPLETS = [
    ("log_bubble_T_arr",    "bubble_T_arr_r_arr",    r"$\log_{10}\,T_{\mathrm{bub}}$"),
    ("log_bubble_n_arr",    "bubble_n_arr_r_arr",    r"$\log_{10}\,n_{\mathrm{bub}}$"),
    ("log_bubble_dTdr_arr", "bubble_dTdr_arr_r_arr", r"$\log_{10}|\,dT/dr|$"),
]

# Phase snapshots laid out top-to-bottom.
_PHASES = ("1_begin", "2_energy", "3_implicit")

_DEFAULT_RUN = "outputs/mockOutput/1e6_sfe001_n1e3_PL0_yesPHII"


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
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
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
                   show: bool = True) -> None:
    """Build the 3×3 diagnostic grid for the given mock-run directory."""
    fig, axes = plt.subplots(
        3, 3, figsize=(13, 10),
        constrained_layout=True,
    )

    for row, phase in enumerate(_PHASES):
        try:
            snap = _load_snapshot(run_dir, phase)
        except FileNotFoundError:
            for col in range(3):
                axes[row, col].set_visible(False)
            continue

        for col, (ykey, xkey, ylabel) in enumerate(_ARRAY_TRIPLETS):
            ax = axes[row, col]
            if xkey not in snap or ykey not in snap:
                ax.set_visible(False)
                continue

            x_raw = np.asarray(snap[xkey], dtype=float)
            y_raw = np.asarray(snap[ykey], dtype=float)

            # Sort ascending for plotting (bubble grids are stored descending).
            order = np.argsort(x_raw)
            x_orig = x_raw[order]
            y_orig = y_raw[order]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                x_simp, y_simp = _simplify(x_orig, y_orig)

            short_y = ykey.replace("_arr", "").replace("log_bubble_", "")
            title = f"{phase} — {short_y}"
            _plot_panel(ax, x_orig, y_orig, x_simp, y_simp,
                        ylabel=ylabel, title=title)

    fig.suptitle(
        f"_simplify diagnostic: pre vs post on {run_dir.name}",
        fontsize=12,
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
        description=("Render a 3×3 grid comparing original and simplified "
                     "bubble profile arrays from a mockOutput run."),
    )
    parser.add_argument(
        "--run", type=Path, default=Path(_DEFAULT_RUN),
        help=f"Mock-run directory (default: {_DEFAULT_RUN})",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="If given, save the figure to this path instead of (or in "
             "addition to) showing it interactively.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not display the figure interactively (useful for batch).",
    )
    args = parser.parse_args(argv)

    run_dir = args.run.resolve()
    if not run_dir.exists():
        parser.error(f"run directory not found: {run_dir}")

    make_diag_grid(run_dir, output=args.output, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
