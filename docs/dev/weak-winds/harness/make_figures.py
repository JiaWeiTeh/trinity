#!/usr/bin/env python3
"""Figures for the weak-winds workstream — reads only committed CSVs.

Usage (from repo root):
    python docs/dev/weak-winds/harness/make_figures.py \
        docs/dev/weak-winds/data/smoke_pair.csv

Writes <csv-stem>_R2.png (R2 + phase marks vs t) and <csv-stem>_forces.png
(driving forces vs t, one panel per run) into docs/dev/weak-winds/figures/.
Style per docs/dev conventions: Agg, no usetex, dpi ~135.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["text.usetex"] = False
FIGDIR = Path(__file__).resolve().parents[1] / "figures"


def load(csv_path):
    runs = defaultdict(list)
    with open(csv_path) as fh:
        rows = csv.DictReader(r for r in fh if not r.startswith("#"))
        for row in rows:
            runs[row["run"]].append(row)
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()
    runs = load(args.csv)
    FIGDIR.mkdir(exist_ok=True)
    stem = args.csv.stem

    # --- R2(t), phase transitions marked ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run, rows in sorted(runs.items()):
        t = [float(r["t_now"]) for r in rows]
        R2 = [float(r["R2"]) for r in rows]
        coeff = rows[0]["FB_thermCoeffWind"]
        (line,) = ax.plot(t, R2, label=f"coeff={coeff}")
        for i in range(1, len(rows)):
            if rows[i]["current_phase"] != rows[i - 1]["current_phase"]:
                ax.axvline(t[i], color=line.get_color(), ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel("R2 [pc]")
    ax.set_title("Shell radius vs wind thermalization (dotted: phase changes)")
    ax.legend()
    fig.tight_layout()
    out = FIGDIR / f"{stem}_R2.png"
    fig.savefig(out, dpi=135)
    print(f"wrote {out}")

    # --- driving forces per run ---
    force_cols = ["F_ram_wind", "F_HII", "F_rad", "F_grav"]
    fig, axes = plt.subplots(
        1, len(runs), figsize=(5.5 * len(runs), 4.2), squeeze=False, sharey=True
    )
    for ax, (run, rows) in zip(axes[0], sorted(runs.items())):
        t = [float(r["t_now"]) for r in rows]
        for col in force_cols:
            ax.plot(t, [abs(float(r[col]) or 0.0) for r in rows], label=col)
        ax.set_yscale("log")
        ax.set_xlabel("t [Myr]")
        ax.set_title(f"coeff={rows[0]['FB_thermCoeffWind']}")
    axes[0][0].set_ylabel("|F| [au]")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    out = FIGDIR / f"{stem}_forces.png"
    fig.savefig(out, dpi=135)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
