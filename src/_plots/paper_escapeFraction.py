#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 15:13:21 2025

@author: Jia Wei Teh
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


print("...plotting escape fraction comparison")


# --- configuration
mCloud_list = ["1e5", "1e7", "1e8"]                 # one subplot per mCloud
ndens_list  = ["1e4", "1e2", "1e3"]                        # one figure per ndens
# ndens_list  = ["1e4"]                        # one figure per ndens
sfe_list    = ["001", "010", "020", "030", "050", "080"]   # multiple lines per subplot

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs"

# smoothing: number of snapshots in moving average (None or 1 disables)
SMOOTH_WINDOW = 7

# --- output
FIG_DIR = Path("./fig")
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PNG = False
SAVE_PDF = True

def range_tag(prefix, values, key=float):
    vals = list(values)
    if len(vals) == 1:
        return f"{prefix}{vals[0]}"
    vmin, vmax = min(vals, key=key), max(vals, key=key)
    return f"{prefix}{vmin}-{vmax}"


def smooth_1d(y, window, mode="edge"):
    """Simple moving-average smoothing. window is in number of snapshots."""
    if window is None or window <= 1:
        return y

    window = int(window)
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode=mode)
    return np.convolve(ypad, kernel, mode="valid")


def load_escape_fraction(json_path: Path):
    """Return (t, fesc) arrays sorted by snapshot index."""
    with json_path.open("r") as f:
        data = json.load(f)

    snap_keys = sorted((k for k in data.keys() if str(k).isdigit()), key=lambda k: int(k))
    snaps = [data[k] for k in snap_keys]

    t = np.array([s["t_now"] for s in snaps], dtype=float)

    # fesc = 1 - fAbs (fAbs stored as shell_fAbsorbedIon)
    fAbs = np.array([s["shell_fAbsorbedIon"] for s in snaps], dtype=float)
    fesc = 1.0 - fAbs
    
    return t, fesc


plt.style.use('/home/user/trinity/src/_plots/trinity.mplstyle')


# --- main plotting: one figure per ndens
for ndens in ndens_list:
    nrows = len(mCloud_list)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=1,
        figsize=(7.0, 2.6 * nrows),
        sharex=False,     # each subplot gets its own t_max
        sharey=True,
        dpi=200,
        constrained_layout=True
    )

    if nrows == 1:
        axes = [axes]  # make iterable

    all_line_handles = []
    all_line_labels = []

    for i, mCloud in enumerate(mCloud_list):
        ax = axes[i]

        # plot each sfe as a line on the same axis
        for sfe in sfe_list:
            run_name = f"{mCloud}_sfe{sfe}_n{ndens}"
            json_path = BASE_DIR / run_name / "dictionary.json"

            if not json_path.exists():
                print(f"Missing: {json_path}")
                continue

            try:
                t, fesc = load_escape_fraction(json_path)

                # optional smoothing
                fesc_plot = smooth_1d(fesc, SMOOTH_WINDOW)
                fesc_plot = np.clip(fesc_plot, 0.0, 1.0)

                eps = int(sfe) / 100.0
                (line,) = ax.plot(t, fesc_plot, lw=1.8, alpha=0.9, label=rf"$\epsilon={eps:.2f}$")

                # store legend handles once (from first subplot only)
                if i == 0:
                    all_line_handles.append(line)
                    all_line_labels.append(rf"$\epsilon={eps:.2f}$")

            except Exception as e:
                print(f"Error in {run_name}: {e}")

        mlog = int(np.log10(float(mCloud)))
        ax.set_ylabel(rf"$f_\mathrm{{esc}}$" + "\n" + rf"$M_{{cloud}}=10^{{{mlog}}}\,M_\odot$")
        ax.set_ylim(0, 1)
        ax.set_xscale('log')

        # x label only on bottom subplot
        if i == nrows - 1:
            ax.set_xlabel("t [Myr]")

    nlog = int(np.log10(float(ndens)))
    fig.suptitle(rf"Escape fraction vs time  ($n=10^{{{nlog}}}\,\mathrm{{cm^{{-3}}}}$)", y=1.02)

    # global legend (cleaner than repeating per axis)
    if all_line_handles:
        leg = fig.legend(
            handles=all_line_handles,
            labels=all_line_labels,
            loc="upper center",
            ncol=len(all_line_handles),
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            edgecolor="0.2",
            bbox_to_anchor=(0.5, 1.07),
        )
        leg.set_zorder(10)

    plt.show()
    plt.close(fig)
