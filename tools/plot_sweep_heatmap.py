#!/usr/bin/env python3
"""Faceted runtime heatmaps for a TRINITY parameter sweep.

Reads the ``sweep_report.json`` written by

    python run.py --collect <jobs_dir>

and renders a grid of *x* x *y* heatmaps coloured by run wall-clock time, one
panel per combination of the remaining swept axes (the "facets"). Runs that
failed (non-zero return code / no sentinel) are masked and drawn grey, so the
map doubles as a picture of *where* in parameter space the sweep broke.

The facets are laid out automatically: the swept axis with the most values
becomes the panel rows, the next becomes the panel columns, and any further
facet axes are split across separate output figures.

Examples
--------
    python tools/plot_sweep_heatmap.py /path/to/sweep_report.json
    python tools/plot_sweep_heatmap.py sweep_report.json --x mCloud --y sfe \
        --out fig/runtime

Only numpy + matplotlib are needed; both ship with the TRINITY environment.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: runs on a login node, writes files only
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

_META = {"duration", "return_code", "success", "error", "output_path", "name"}


def _num(v):
    """Best-effort float (for sorting/formatting); falls back to the raw value."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _fmt(v):
    """Compact axis-tick label: 1e5, 5e9, 0.01, 0 ..."""
    f = _num(v)
    if not isinstance(f, float):
        return str(v)
    if f == 0:
        return "0"
    if abs(f) >= 1e4 or abs(f) < 1e-2:
        mant, exp = f"{f:.0e}".split("e")
        return f"{mant}e{int(exp)}"
    return f"{f:g}"


def _sort_key(v):
    n = _num(v)
    return (0, n) if isinstance(n, float) else (1, str(v))


def load_results(report_path):
    """sweep_report.json -> list of flat dicts (params + duration/return_code)."""
    data = json.loads(Path(report_path).read_text())
    rows = []
    for r in data.get("results", []):
        row = dict(r.get("params", {}))
        row["duration"] = r.get("duration")
        row["return_code"] = r.get("return_code")
        row["success"] = r.get("success")
        rows.append(row)
    if not rows:
        raise SystemExit(f"No results found in {report_path}.")
    return rows


def unique_sorted(rows, key):
    return sorted({r[key] for r in rows if r.get(key) is not None}, key=_sort_key)


def facet_axes(rows, x_key, y_key):
    """Swept axes other than x/y, ordered by descending number of values."""
    keys = set().union(*(r.keys() for r in rows)) - _META - {x_key, y_key}
    varying = [k for k in keys if len(unique_sorted(rows, k)) > 1]
    return sorted(varying, key=lambda k: (-len(unique_sorted(rows, k)), k))


def runtime_grid(index, x_key, y_key, x_vals, y_vals, facet_keys, facet_vals):
    """(len(y_vals), len(x_vals)) array of durations; NaN where failed/missing."""
    z = np.full((len(y_vals), len(x_vals)), np.nan)
    for yi, yv in enumerate(y_vals):
        for xi, xv in enumerate(x_vals):
            rec = index.get((xv, yv) + facet_vals)
            if rec is None:
                continue
            dur = rec.get("duration")
            if rec.get("success") and dur not in (None, 0):
                z[yi, xi] = dur / 60.0  # seconds -> minutes
    return z


def _panel(ax, z, x_vals, y_vals, norm, cmap):
    mesh = ax.pcolormesh(
        np.arange(len(x_vals) + 1), np.arange(len(y_vals) + 1),
        np.ma.masked_invalid(z), norm=norm, cmap=cmap, shading="flat",
    )
    ax.set_xticks(np.arange(len(x_vals)) + 0.5)
    ax.set_yticks(np.arange(len(y_vals)) + 0.5)
    ax.set_xticklabels([_fmt(v) for v in x_vals], rotation=90, fontsize=6)
    ax.set_yticklabels([_fmt(v) for v in y_vals], fontsize=6)
    ax.tick_params(length=0)
    return mesh


def make_figures(rows, x_key, y_key, out_prefix, vmin=10.0, vmax=120.0):
    x_vals = unique_sorted(rows, x_key)
    y_vals = unique_sorted(rows, y_key)
    facets = facet_axes(rows, x_key, y_key)
    index = {
        (r[x_key], r[y_key]) + tuple(r.get(f) for f in facets): r
        for r in rows
        if r.get(x_key) is not None and r.get(y_key) is not None
    }

    durations = [r["duration"] for r in rows
                 if r.get("success") and r.get("duration") not in (None, 0)]
    if not durations:
        raise SystemExit("No successful runs with a duration to plot.")
    norm = LogNorm(vmin=vmin, vmax=vmax)  # fixed caps, in minutes
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.85")  # grey for failed / missing cells

    # Split facets into rows / cols / per-figure axes.
    row_key = facets[0] if facets else None
    col_key = facets[1] if len(facets) > 1 else None
    file_keys = facets[2:]
    row_vals = unique_sorted(rows, row_key) if row_key else [None]
    col_vals = unique_sorted(rows, col_key) if col_key else [None]
    file_val_lists = [unique_sorted(rows, k) for k in file_keys]

    n_fail = sum(1 for r in rows if not r.get("success"))
    written = []
    for file_combo in (itertools.product(*file_val_lists) or [()]):
        nrows, ncols = len(row_vals), len(col_vals)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(2.4 * ncols + 1.5, 2.2 * nrows + 1.0),
            squeeze=False, constrained_layout=True,
        )
        mesh = None
        for ri, rv in enumerate(row_vals):
            for ci, cv in enumerate(col_vals):
                ax = axes[ri][ci]
                facet_vals = tuple(
                    v for v, k in ((rv, row_key), (cv, col_key)) if k is not None
                ) + tuple(file_combo)
                z = runtime_grid(index, x_key, y_key, x_vals, y_vals,
                                 facets, facet_vals)
                mesh = _panel(ax, z, x_vals, y_vals, norm, cmap)
                bits = []
                if col_key:
                    bits.append(f"{col_key}={_fmt(cv)}")
                if row_key:
                    bits.append(f"{row_key}={_fmt(rv)}")
                ax.set_title("  ".join(bits), fontsize=7)
                if ci == 0:
                    ax.set_ylabel(y_key, fontsize=8)
                if ri == nrows - 1:
                    ax.set_xlabel(x_key, fontsize=8)

        sub = "  ".join(f"{k}={_fmt(v)}" for k, v in zip(file_keys, file_combo))
        title = "TRINITY sweep runtime"
        if sub:
            title += f"   ({sub})"
        title += f"    [{len(durations)} ok, {n_fail} failed — grey = failed/missing]"
        fig.suptitle(title, fontsize=10)
        cbar = fig.colorbar(mesh, ax=axes, shrink=0.6, pad=0.01, extend="both")
        cbar.set_label("wall-clock runtime [min]", fontsize=8)

        suffix = "_" + "_".join(f"{k}{_fmt(v)}" for k, v in zip(file_keys, file_combo))
        out = Path(f"{out_prefix}{suffix if sub else ''}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", help="path to sweep_report.json")
    ap.add_argument("--x", default="sfe", help="x-axis param (default: sfe)")
    ap.add_argument("--y", default="mCloud", help="y-axis param (default: mCloud)")
    ap.add_argument("--vmin", type=float, default=10.0,
                    help="colourbar minimum, in minutes (default: 10)")
    ap.add_argument("--vmax", type=float, default=120.0,
                    help="colourbar maximum, in minutes (default: 120 = 2 h)")
    ap.add_argument("--out", default=None,
                    help="output prefix (default: <report dir>/runtime_heatmap)")
    args = ap.parse_args()

    rows = load_results(args.report)
    out_prefix = args.out or str(Path(args.report).resolve().parent / "runtime_heatmap")
    written = make_figures(rows, args.x, args.y, out_prefix,
                           vmin=args.vmin, vmax=args.vmax)
    print(f"Wrote {len(written)} figure(s):")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
