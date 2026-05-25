#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the time evolution of important parameters between two TRINITY runs.

Takes two output folders (each containing a ``dictionary.jsonl``) and
produces three PDF grid plots — one per category:

  - dynamics : radii, velocity, masses, energy, temperature, pressure
  - sps      : stellar-population feedback (Qi, Lmech, Lbol, pdot, ...)
  - forces   : feedback terms acting on the shell (F_grav, F_HII, ...)

Usage
-----
  python -m src._plots.compare_outputs <folderA> <folderB> [options]

Each <folder> may be:
  - a run folder containing ``dictionary.jsonl``
  - a path directly to a .jsonl file
  - a simulation name resolvable inside ``outputs/``
"""

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src._plots.plot_base import FIG_DIR  # noqa: E402  applies trinity.mplstyle
from src._output.trinity_reader import (  # noqa: E402
    TrinityOutput, resolve_data_input,
)


# ---------------------------------------------------------------------------
# Parameter groupings
# ---------------------------------------------------------------------------
# Each entry: (key, latex_label, y_scale)  --  y_scale in {"linear", "log", "symlog"}

DYNAMICS_PARAMS = [
    ("R1",          r"$R_1$ [pc]",                "log"),
    ("R2",          r"$R_2$ [pc]",                "log"),
    ("rShell",      r"$r_{\rm shell}$ [pc]",      "log"),
    ("v2",          r"$v_2$ [pc/Myr]",            "log"),
    ("shell_mass",  r"$M_{\rm shell}$ [M$_\odot$]", "log"),
    ("bubble_mass", r"$M_{\rm bubble}$ [M$_\odot$]", "log"),
    ("Eb",          r"$E_b$ [erg]",               "log"),
    ("T0",          r"$T_0$ [K]",                 "log"),
    ("Pb",          r"$P_b$ [dyn/cm$^2$]",        "log"),
]

SPS_PARAMS = [
    ("Qi",           r"$Q_i$ [photons/s]",        "log"),
    ("Lbol",         r"$L_{\rm bol}$ [code]",     "log"),
    ("Li",           r"$L_i$ [code]",             "log"),
    ("Ln",           r"$L_n$ [code]",             "log"),
    ("Lmech_W",      r"$L_{\rm mech,W}$ [code]",  "log"),
    ("Lmech_SN",     r"$L_{\rm mech,SN}$ [code]", "log"),
    ("Lmech_total",  r"$L_{\rm mech}$ [code]",    "log"),
    ("pdot_W",       r"$\dot p_W$",               "log"),
    ("pdot_total",   r"$\dot p_{\rm tot}$",       "log"),
]

FORCE_PARAMS = [
    ("F_grav",   r"$F_{\rm grav}$",         "log"),
    ("F_ram",    r"$F_{\rm ram}$",          "log"),
    ("F_HII",    r"$F_{\rm HII}$",          "log"),
    ("F_rad",    r"$F_{\rm rad}$",          "log"),
    ("F_ion_in", r"$F_{\rm ion,in}$",       "log"),
    ("F_ISM",    r"$F_{\rm ISM}$",          "log"),
    ("P_HII",    r"$P_{\rm HII}$",          "log"),
    ("P_drive",  r"$P_{\rm drive}$",        "log"),
    ("P_ram",    r"$P_{\rm ram}$",          "log"),
]

CATEGORIES = {
    "dynamics": DYNAMICS_PARAMS,
    "sps":      SPS_PARAMS,
    "forces":   FORCE_PARAMS,
}

# Visual style for the two runs
RUN_STYLES = [
    {"color": "#1f77b4", "ls": "-",  "lw": 1.7},  # blue, solid  -- run A
    {"color": "#d62728", "ls": "--", "lw": 1.7},  # red,  dashed -- run B
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _to_float_array(values):
    """Convert a snapshot list to a float ndarray, mapping non-numeric to NaN."""
    out = np.full(len(values), np.nan, dtype=float)
    for i, v in enumerate(values):
        if v is None:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[i] = float(v)
    return out


def load_run(folder):
    """Resolve *folder* to a ``dictionary.jsonl`` and return a TrinityOutput."""
    path = resolve_data_input(folder)
    return TrinityOutput.open(path)


def time_series(output: TrinityOutput, key: str):
    """Return (t, y) with non-finite samples masked, sorted by time."""
    t = _to_float_array(output.get('t_now', as_array=False))
    if key not in output.keys:
        return None, None
    y = _to_float_array(output.get(key, as_array=False))
    if t.size != y.size:
        return None, None
    order = np.argsort(t)
    t, y = t[order], y[order]
    mask = np.isfinite(t) & np.isfinite(y)
    return t[mask], y[mask]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _grid_shape(n):
    """Pick (nrows, ncols) keeping the grid roughly square."""
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols


def _safe_log_data(y):
    """Mask non-positive samples so log scale doesn't choke."""
    return np.where(y > 0, y, np.nan)


def plot_category(
    category_name,
    params,
    outputs,
    labels,
    out_path,
    use_log_x=False,
    title_extra="",
):
    """Render one grid plot for a single category and save as PDF."""
    nrows, ncols = _grid_shape(len(params))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(3.6 * ncols, 2.7 * nrows),
        dpi=150, squeeze=False,
    )

    for idx, (key, label, yscale) in enumerate(params):
        i, j = divmod(idx, ncols)
        ax = axes[i, j]

        any_data = False
        for run_idx, output in enumerate(outputs):
            t, y = time_series(output, key)
            if t is None or t.size == 0:
                continue
            if yscale == "log":
                y_plot = _safe_log_data(np.abs(y))
            else:
                y_plot = y
            style = RUN_STYLES[run_idx]
            ax.plot(
                t, y_plot,
                color=style["color"], ls=style["ls"], lw=style["lw"],
                label=labels[run_idx],
            )
            any_data = True

        if not any_data:
            ax.text(0.5, 0.5, f"{key}\n(not present)",
                    ha="center", va="center", transform=ax.transAxes,
                    color="grey", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        ax.set_ylabel(label, fontsize=10)
        if yscale == "log":
            ax.set_yscale("log")
        if use_log_x:
            ax.set_xscale("log")
        if i == nrows - 1:
            ax.set_xlabel("t [Myr]")
        ax.tick_params(labelsize=8)
        ax.grid(True, which="both", alpha=0.25, lw=0.5)

    # Blank out any unused cells in the bottom row
    for idx in range(len(params), nrows * ncols):
        i, j = divmod(idx, ncols)
        axes[i, j].axis("off")

    # Shared legend up top
    legend_handles = [
        Line2D([0], [0],
               color=RUN_STYLES[k]["color"],
               ls=RUN_STYLES[k]["ls"],
               lw=RUN_STYLES[k]["lw"],
               label=labels[k])
        for k in range(len(outputs))
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center", ncol=len(outputs),
        bbox_to_anchor=(0.5, 0.995), frameon=False,
        fontsize=10,
    )

    suptitle = f"Comparison: {category_name}"
    if title_extra:
        suptitle += f"  ({title_extra})"
    fig.suptitle(suptitle, y=0.965, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare time evolution of important parameters between "
                    "two TRINITY output folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src._plots.compare_outputs outputs/runA outputs/runB
  python -m src._plots.compare_outputs runA runB --labels baseline modified
  python -m src._plots.compare_outputs A B --log-x --output-dir fig/compare
        """,
    )
    parser.add_argument("folderA", help="First output folder / file / run name")
    parser.add_argument("folderB", help="Second output folder / file / run name")
    parser.add_argument(
        "--labels", nargs=2, metavar=("LABEL_A", "LABEL_B"), default=None,
        help="Legend labels for the two runs (default: folder names)",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help=f"Output directory for PDFs (default: {FIG_DIR})",
    )
    parser.add_argument(
        "--log-x", action="store_true",
        help="Use log scale on the time axis.",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Optional suffix appended to output PDF filenames.",
    )

    args = parser.parse_args()

    output_a = load_run(args.folderA)
    output_b = load_run(args.folderB)
    outputs = [output_a, output_b]

    if args.labels:
        labels = args.labels
    else:
        labels = [
            output_a.filepath.parent.name or output_a.filepath.stem,
            output_b.filepath.parent.name or output_b.filepath.stem,
        ]

    out_dir = Path(args.output_dir) if args.output_dir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"_{args.tag}" if args.tag else ""
    title_extra = f"{labels[0]}  vs  {labels[1]}"

    for cat_name, params in CATEGORIES.items():
        out_path = out_dir / f"compare_{cat_name}_{labels[0]}_vs_{labels[1]}{tag}.pdf"
        plot_category(
            category_name=cat_name,
            params=params,
            outputs=outputs,
            labels=labels,
            out_path=out_path,
            use_log_x=args.log_x,
            title_extra=title_extra,
        )


if __name__ == "__main__":
    main()
