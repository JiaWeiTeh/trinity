#!/usr/bin/env python3
"""Stock vs C3c on one canvas, using paper_feedback's OWN loader and renderer.

The point of this script is that it adds no plotting logic: it imports
``load_run`` and ``plot_run_on_ax`` from ``paper/methods/figures/paper_feedback.py``
and calls them once per arm. Whatever the published force-fraction figure would
show for each run is what appears here, so the difference on screen is a
difference in the runs, never in the drawing.

Usage:
    python docs/dev/phii-identity/harness/plot_phii_feedback_compare.py \
        --stock <run_dir> --c3c <run_dir> --out fig/phii_feedback_compare
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "paper" / "methods" / "figures"))

import paper_feedback as pf  # noqa: E402
from trinity._output.trinity_reader import resolve_data_input  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", required=True)
    ap.add_argument("--c3c", required=True)
    ap.add_argument("--out", required=True, help="output basename (no extension)")
    ap.add_argument("--title", default="B3M")
    args = ap.parse_args()

    arms = [("stock — capped Strömgren $P_{\\rm HII}$", args.stock),
            ("C3c — confinement regime switch", args.c3c)]

    loaded = [(label, pf.load_run(resolve_data_input(run_dir))) for label, run_dir in arms]

    # Matched t (CLAUDE.md rule 5): the arms truncate at different simulation
    # times, so compare only over the window both actually reached. Without this
    # the eye reads "C3c evolves further" when it has merely run longer.
    t_hi = min(d[0].max() for _, d in loaded)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), dpi=170, sharey=True)
    for ax, (label, data) in zip(axes, loaded):
        t, R2, phase, base_f, over_f, rcloud, isColl, press = data
        m = t <= t_hi
        t, R2, phase = t[m], R2[m], phase[m]
        base_f, over_f, press = base_f[:, m], over_f[:, m], press[:, m]
        pf.plot_run_on_ax(
            ax, t, R2, phase, base_f, over_f, rcloud, isColl,
            pressures=press, alpha=0.75, smooth_window=pf.SMOOTH_WINDOW,
            phase_change=True, use_log_x=pf.USE_LOG_X,
        )
        ax.set_xlabel("t [Myr]")
        ax.set_title(label, fontsize=10)
    axes[0].set_ylabel(r"$F/F_{\rm tot}$")

    handles = [
        Patch(facecolor=pf.C_GRAV, alpha=0.75, label="Gravity"),
        Patch(facecolor=pf.C_DRIVE, alpha=0.75, label=r"$F_{\rm drive}$"),
        Patch(facecolor=pf.C_RAD, alpha=0.75, label="Radiation"),
        Patch(facecolor=pf.C_PISM, edgecolor="0.3", lw=0.8, label="PISM (inner HII)"),
        Patch(facecolor="none", edgecolor=pf.C_PHII, hatch="......", label=r"$P_{\rm HII}$"),
        Patch(facecolor="none", edgecolor=pf.C_WIND, hatch="\\\\\\\\", label="Ram wind"),
        Patch(facecolor="none", edgecolor=pf.C_SN, hatch="////", label="Ram SN"),
        Line2D([0], [0], color=pf.C_PHII, lw=3, alpha=0.7, label=r"Driver: $P_{\rm HII}$"),
        Line2D([0], [0], color=pf.C_DRIVE, lw=3, alpha=0.7, label=r"Driver: $P_b$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(f"force-fraction budget, {args.title}: what C3c changed  "
                 f"(matched $t \\leq {t_hi:.4g}$ Myr)", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"wrote {out}.png / .pdf")


if __name__ == "__main__":
    main()
