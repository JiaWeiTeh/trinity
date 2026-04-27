#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHII active-window diagnostic.

Answers the question: why do include_PHII=True (yesPHII) and include_PHII=False
(noPHII) runs produce nearly identical R(t)?

The implicit-phase rhs drives the shell with P_drive = max(Pb, P_HII). So
include_PHII can only change the dynamics in the time window where
P_HII > Pb. Outside that window, the two ODEs are bit-identical.

For each matched _yesPHII / _noPHII pair (auto-paired by stripped base name,
same convention as paper_radiusComparison.py), this script draws a 2-panel
figure stacked on a shared time axis:

    Top:    Pb(t) and P_HII(t) from the yesPHII run, with the PHII-active
            window (P_HII > Pb) shaded across both panels.
    Bottom: |v_2(t)| from yesPHII (black) and noPHII (red dashed). Shows
            the divergence inside the active window and the convergence
            once Pb takes over.

One PDF per pair, saved to FIG_DIR/<folder_name>/.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src._plots.plot_base import FIG_DIR, smooth_1d
from src._output.trinity_reader import (
    load_output, find_all_simulations, organize_simulations_for_grid,
    get_unique_ndens,
)
from src._functions.unit_conversions import INV_CONV, CGS

print("...plotting PHII active-window diagnostic")

# Pressure conversion: code units (Msun/pc/Myr^2) -> P/k_B [K cm^-3]
P_AU_TO_K_CM3 = INV_CONV.Pb_au2cgs / CGS.k_B

YES_SUFFIX = "_yesPHII"
NO_SUFFIX = "_noPHII"

# Styling
COLOR_PB    = "tab:blue"
COLOR_PHII  = "tab:red"
COLOR_YES   = "k"
COLOR_NO    = "tab:red"
COLOR_BAND  = "0.85"   # PHII-active shading
ALPHA_BAND  = 0.45

LW_PRESSURE = 1.6
LW_VEL      = 1.6

SMOOTH_WINDOW = None   # set to e.g. 5 to smooth noisy outputs


def load_run(data_path):
    """Load t, R2, v2, Pb, P_HII for one run, time-sorted."""
    output = load_output(data_path)
    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    def get_field(field, default=np.nan):
        arr = output.get(field)
        if arr is None or (isinstance(arr, np.ndarray) and np.all(arr == None)):
            return np.full(len(output), default)
        return np.where(arr == None, default, arr).astype(float)

    t  = output.get('t_now')
    R2 = output.get('R2')
    v2 = get_field('v2', np.nan)
    Pb    = get_field('Pb', np.nan) * P_AU_TO_K_CM3
    P_HII = get_field('P_HII', np.nan) * P_AU_TO_K_CM3
    rcloud = float(output[0].get('rCloud', np.nan))

    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, R2, v2, Pb, P_HII = (a[order] for a in (t, R2, v2, Pb, P_HII))

    return dict(t=t, R2=R2, v2=v2, Pb=Pb, P_HII=P_HII, rcloud=rcloud)


def find_active_segments(t, P_HII, Pb):
    """Return list of (t_start, t_end) where P_HII > Pb (and both finite)."""
    mask = np.isfinite(P_HII) & np.isfinite(Pb) & (P_HII > Pb)
    if not np.any(mask):
        return []

    segs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            segs.append((t[i], t[max(j - 1, i)]))
            i = j
        else:
            i += 1
    return segs


def shade_active_window(ax, segments):
    for (t0, t1) in segments:
        ax.axvspan(t0, t1, color=COLOR_BAND, alpha=ALPHA_BAND, zorder=0,
                   linewidth=0)


def plot_pair(data_yes, data_no, base_name, out_pdf):
    """Two-panel figure: pressure crossover (top) + |v2| comparison (bottom)."""
    fig, (ax_p, ax_v) = plt.subplots(
        nrows=2, ncols=1, sharex=True,
        figsize=(4.6, 4.4), dpi=300,
        gridspec_kw=dict(hspace=0.05, height_ratios=[1.0, 1.0]),
    )

    t_y  = data_yes['t']
    Pb_y    = smooth_1d(data_yes['Pb'],    SMOOTH_WINDOW)
    P_HII_y = smooth_1d(data_yes['P_HII'], SMOOTH_WINDOW)
    v2_y = smooth_1d(np.abs(data_yes['v2']), SMOOTH_WINDOW)

    t_n  = data_no['t']
    v2_n = smooth_1d(np.abs(data_no['v2']), SMOOTH_WINDOW)

    # PHII active window from the yesPHII run (since that is where
    # max(Pb, P_HII) actually selects P_HII)
    segments = find_active_segments(t_y, P_HII_y, Pb_y)

    # Top: pressures
    shade_active_window(ax_p, segments)
    ax_p.plot(t_y, Pb_y,    color=COLOR_PB,   lw=LW_PRESSURE, label=r"$P_b$")
    ax_p.plot(t_y, P_HII_y, color=COLOR_PHII, lw=LW_PRESSURE, label=r"$P_{\rm HII}$")
    ax_p.set_yscale('log')
    ax_p.set_ylabel(r"$P/k_B$  [K cm$^{-3}$]")
    ax_p.legend(loc="upper right", frameon=False, fontsize=9)

    # Bottom: |v2|
    shade_active_window(ax_v, segments)
    ax_v.plot(t_y, v2_y, color=COLOR_YES, lw=LW_VEL, ls="-",  label="yesPHII")
    ax_v.plot(t_n, v2_n, color=COLOR_NO,  lw=LW_VEL, ls="--", label="noPHII")
    ax_v.set_yscale('log')
    ax_v.set_ylabel(r"$|v_2|$  [pc Myr$^{-1}$]")
    ax_v.set_xlabel(r"$t$  [Myr]")
    ax_v.legend(loc="upper right", frameon=False, fontsize=9)

    # Title: base run name + total active-window duration
    if segments:
        active_dt = sum(t1 - t0 for (t0, t1) in segments)
        active_label = f"PHII-active: {active_dt*1000:.1f} kyr in {len(segments)} window(s)"
    else:
        active_label = r"PHII-active: never ($P_{\rm HII}\le P_b$ throughout)"
    fig.suptitle(f"{base_name}\n{active_label}", fontsize=9, y=0.995)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_pdf}")


def split_by_phii_suffix(folder):
    """Split simulations in folder into yesPHII paths + noPHII lookup."""
    yes_paths = []
    no_by_base = {}
    for p in find_all_simulations(folder):
        name = p.parent.name
        if name.endswith(YES_SUFFIX):
            yes_paths.append(p)
        elif name.endswith(NO_SUFFIX):
            no_by_base[name[: -len(NO_SUFFIX)]] = p
    return yes_paths, no_by_base


def filter_pairs(yes_paths, ndens=None, mClouds=None, sfes=None):
    """Keep only yes paths whose tags pass the filters."""
    organized = organize_simulations_for_grid(
        yes_paths, ndens_filter=ndens,
        mCloud_filter=mClouds, sfe_filter=sfes,
    )
    grid = organized['grid']
    return list(grid.values())


def run(folder, output_dir=None, ndens=None, mClouds=None, sfes=None):
    folder = Path(folder)
    yes_paths, no_by_base = split_by_phii_suffix(folder)
    if not yes_paths:
        print(f"No _yesPHII simulations found in: {folder}")
        return

    if ndens is None:
        densities = get_unique_ndens(yes_paths)
    else:
        densities = [ndens]

    fig_dir = Path(output_dir) if output_dir else FIG_DIR / folder.name
    fig_dir.mkdir(parents=True, exist_ok=True)

    for n in densities:
        kept = filter_pairs(yes_paths, ndens=n, mClouds=mClouds, sfes=sfes)
        print(f"\nn={n}: {len(kept)} yesPHII run(s) after filter")
        for path_y in kept:
            base = path_y.parent.name[: -len(YES_SUFFIX)]
            path_n = no_by_base.get(base)
            if path_n is None:
                print(f"  [skip] no _noPHII partner for {base}")
                continue
            try:
                data_y = load_run(path_y)
                data_n = load_run(path_n)
            except Exception as e:
                print(f"  [error] {base}: {e}")
                continue

            out_pdf = fig_dir / f"phii_window_{base}.pdf"
            plot_pair(data_y, data_n, base, out_pdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show the PHII-active window (P_HII > Pb) and the "
                    "yes/no PHII velocity comparison.",
    )
    parser.add_argument('--folder', '-F', required=True,
                        help='Folder containing _yesPHII and _noPHII subfolders')
    parser.add_argument('--output-dir', '-o', default=None)
    parser.add_argument('--nCore', '-n', default=None,
                        help='Filter by density tag (e.g. 1e3)')
    parser.add_argument('--mCloud', nargs='+', default=None)
    parser.add_argument('--sfe', nargs='+', default=None)
    args = parser.parse_args()

    run(
        args.folder,
        output_dir=args.output_dir,
        ndens=args.nCore,
        mClouds=args.mCloud,
        sfes=args.sfe,
    )
