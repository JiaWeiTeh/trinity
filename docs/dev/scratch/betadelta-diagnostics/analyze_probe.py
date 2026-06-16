#!/usr/bin/env python3
"""(beta, delta) phase-space maps from the Phase-2 probe sweeps.

Reads the probe_*.jsonl written by probe.py (re-readable scratch data, not
ephemeral). Each probed segment carries a regular 7x7 ``scan`` grid over
beta in [-1, 2] x delta in [-1, 0.5] -- every grid point recording the two
residual components (f_E, f_T), the detailed pieces (E1=Edot_from_beta,
E2=Edot_from_balance, Lg=L_gain=Lmech_total), feasibility (``ok`` = structure
integrator returned a solution), and the dMdt>0 acceptance flag (``dmdt_ok``)
-- plus the solver's accepted root (``accept``).

Two residual metrics (both: total = E-comp**2 + T-comp**2, converged < 1e-4):
  f (legacy) : f_E = (E1-E2)/E1   -- denominator E1 hits 0 near the E_b peak,
               a pole that wrecks the bounded solver.
  g (hybr)   : g_E = (E1-E2)/Lg   -- per-segment-constant denominator, pole-free
               (reconstructed here from the recorded E1, E2, Lg -- no rerun).

Legacy clamp box = beta in [0, 1], delta in [-1, 0]; the scan reaches outside it.
Per config the segment shown is the one whose accepted root has the lowest
total_residual -- the probe's *f*-metric (pure-solver) total; that pure-solver
root is the white star on the maps (a reference point, NOT the g-minimum).
Produces:
  - betadelta_gmap.png   : g-metric residual + feasibility, one panel/config (2x2)
  - betadelta_f_vs_g.png : f vs g side by side (typical converger, steep staller)

Usage: python docs/dev/scratch/betadelta-diagnostics/analyze_probe.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
RES_THRESH = 1e-4  # RESIDUAL_THRESHOLD
BETA_BOX, DELTA_BOX = (0.0, 1.0), (-1.0, 0.0)  # legacy clamp
BETA_AX = np.linspace(-1.0, 2.0, 7)  # SCAN_BETA from probe.py
DELTA_AX = np.linspace(-1.0, 0.5, 7)  # SCAN_DELTA
DB, DD = 0.5, 0.25  # grid spacing (for per-cell hatch rectangles)

# (file, short label) -- segment auto-picked as the lowest-residual accept
CONFIGS = [
    ("probe_cloud1e6.jsonl", "typical 1e6, n=1e3, α=0"),  # nCore=1e3 (NOT the dense flat n=1e5)
    ("probe_cloudPL.jsonl", "steep  1e6, α=−2, n=1e5"),
    ("probe_mock4e3.jsonl", "mock   4e3, low-mass"),
    ("probe_simple1e5.jsonl", "simple 1e5, sfe=0.3"),
]

_STYLE = HERE.parents[1] / "paper" / "_lib" / "trinity.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))
plt.rcParams["text.usetex"] = False


def load(fn):
    return [json.loads(line) for line in open(HERE / fn) if line.strip()]


def pick_segment(recs):
    """Segment whose accepted root has the smallest (f-metric) total_residual."""
    accepts = [r for r in recs if r["kind"] == "accept"]
    best = min(accepts, key=lambda r: r["total_residual"])
    return best["segment"], best


def _idx(r):
    i = int(np.argmin(np.abs(BETA_AX - r["beta"])))
    j = int(np.argmin(np.abs(DELTA_AX - r["delta"])))
    return i, j


def g_grid(recs, seg):
    """7x7 g-metric residual grid + the dMdt<=0 abort cells.

    g_total = ((E1-E2)/Lg)**2 + f_T**2 over cells with a bubble structure;
    no-structure cells stay NaN (masked gray). ``abort`` lists (i, j) cells that
    have a structure but dMdt<=0 -- where the hybr acceptance gate rejects.
    """
    g = np.full((7, 7), np.nan)
    abort = []
    for r in recs:
        if r["kind"] != "scan" or r["segment"] != seg or not r.get("ok"):
            continue
        i, j = _idx(r)
        g[j, i] = ((r["E1"] - r["E2"]) / r["Lg"]) ** 2 + r["f_T"] ** 2
        if not r.get("dmdt_ok"):
            abort.append((i, j))
    return g, abort


def f_and_g(recs, seg):
    """7x7 f-metric and g-metric grids for the f-vs-g comparison."""
    fgrid = np.full((7, 7), np.nan)
    ggrid = np.full((7, 7), np.nan)
    for r in recs:
        if r["kind"] != "scan" or r["segment"] != seg or not r.get("ok"):
            continue
        i, j = _idx(r)
        fgrid[j, i] = r["f_E"] ** 2 + r["f_T"] ** 2
        ggrid[j, i] = ((r["E1"] - r["E2"]) / r["Lg"]) ** 2 + r["f_T"] ** 2
    return fgrid, ggrid


def _clampbox(ax):
    ax.add_patch(
        Rectangle(
            (BETA_BOX[0], DELTA_BOX[0]),
            BETA_BOX[1] - BETA_BOX[0],
            DELTA_BOX[1] - DELTA_BOX[0],
            fill=False,
            edgecolor="cyan",
            lw=2.0,
            zorder=5,
        )
    )


def _root(ax, acc):
    ax.plot(
        acc["beta"],
        acc["delta"],
        marker="*",
        ms=18,
        mfc="white",
        mec="k",
        mew=1.2,
        ls="none",
        zorder=6,
    )


def _hatch_abort(ax, abort):
    for i, j in abort:
        ax.add_patch(
            Rectangle(
                (BETA_AX[i] - DB / 2, DELTA_AX[j] - DD / 2),
                DB,
                DD,
                fill=False,
                hatch="xxxx",
                edgecolor="0.15",
                lw=0.0,
                zorder=4,
            )
        )


def _axfmt(ax):
    ax.set_xlim(-1.3, 2.3)
    ax.set_ylim(-1.2, 0.7)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\delta$")


def plot_gmap(data, path):
    """One figure: g-metric residual + feasibility, a panel per config."""
    allg = np.concatenate([d[1][np.isfinite(d[1])] for d in data])
    norm = LogNorm(vmin=max(allg.min(), 1e-4), vmax=allg.max())
    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad("0.82")  # no-structure cells

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), constrained_layout=True)
    for ax, (label, g, abort, acc) in zip(axes.flat, data):
        m = ax.pcolormesh(
            BETA_AX, DELTA_AX, np.ma.masked_invalid(g), norm=norm, cmap=cmap, shading="nearest"
        )
        _hatch_abort(ax, abort)
        _clampbox(ax)
        _root(ax, acc)
        _axfmt(ax)
        ax.set_title(f"{label}\nseg {acc['segment']}, t={acc['t_now']:.3g} Myr", fontsize=9)
    cb = fig.colorbar(m, ax=axes, shrink=0.85, label=r"$g$ residual  $g_E^2+g_T^2$  (log)")
    cb.ax.axhline(RES_THRESH, color="white", lw=1.4)
    handles = [
        plt.Line2D([], [], marker="*", mfc="white", mec="k", ls="none", ms=12, label="root (ref)"),
        Patch(fill=False, edgecolor="cyan", lw=2, label="legacy clamp box"),
        Patch(facecolor="white", hatch="xxxx", edgecolor="0.15", label="structure, dMdt≤0 (abort)"),
        Patch(facecolor="0.82", label="no structure"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.02)
    )
    fig.suptitle(
        r"(β, δ) g-metric (pole-free) residual + feasibility  (conv threshold "
        f"{RES_THRESH:g} on bar)",
        fontsize=12,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_f_vs_g(items, path):
    """Rows = config, cols = metric (f legacy vs g hybr), shared colour scale."""
    norm = LogNorm(vmin=1e-3, vmax=1e1)  # f-pole saturates at top; g stays low
    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad("0.82")
    col_titles = ["f-metric (legacy) — pole at Edot_β=0", "g-metric (hybr) — pole-free"]

    fig, axes = plt.subplots(len(items), 2, figsize=(11, 5 * len(items)), constrained_layout=True)
    for row, (label, fgrid, ggrid, acc) in enumerate(items):
        for col, grid in enumerate((fgrid, ggrid)):
            ax = axes[row, col]
            m = ax.pcolormesh(
                BETA_AX,
                DELTA_AX,
                np.ma.masked_invalid(grid),
                norm=norm,
                cmap=cmap,
                shading="nearest",
            )
            _clampbox(ax)
            _root(ax, acc)
            _axfmt(ax)
            if col == 0:
                ax.set_ylabel(f"{label}\n\n" + r"$\delta$")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11)
    cb = fig.colorbar(m, ax=axes, shrink=0.8, label="residual  (sum of squares, log)")
    cb.ax.axhline(RES_THRESH, color="white", lw=1.4)
    handles = [
        Patch(fill=False, edgecolor="cyan", lw=2, label="legacy clamp box"),
        plt.Line2D(
            [],
            [],
            marker="*",
            mfc="white",
            mec="k",
            ls="none",
            ms=12,
            label="pure-solver root (ref)",
        ),
        Patch(facecolor="0.82", label="no structure"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.03)
    )
    fig.suptitle(
        "Same scan, two metrics — the f-pole (dark stripe at β≈1.5) vanishes under g", fontsize=12
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    gdata = []
    for fn, label in CONFIGS:
        recs = load(fn)
        seg, acc = pick_segment(recs)
        g, abort = g_grid(recs, seg)
        gdata.append((label, g, abort, acc))
    plot_gmap(gdata, HERE / "betadelta_gmap.png")

    fg = []
    for fn, label in (CONFIGS[0], CONFIGS[1]):  # typical converger, steep staller
        recs = load(fn)
        seg, acc = pick_segment(recs)
        fgrid, ggrid = f_and_g(recs, seg)
        fg.append((label, fgrid, ggrid, acc))
    plot_f_vs_g(fg, HERE / "betadelta_f_vs_g.png")
    print("wrote betadelta_gmap.png, betadelta_f_vs_g.png")


if __name__ == "__main__":
    main()
