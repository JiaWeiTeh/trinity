#!/usr/bin/env python3
"""(beta, delta) phase-space maps from the Phase-2 probe sweeps.

Reads the probe_*.jsonl written by probe.py (re-readable scratch data, not
ephemeral). Each probed segment carries a regular 7x7 ``scan`` grid over
beta in [-1, 2] x delta in [-1, 0.5] -- every grid point recording the two
residual components (f_E, f_T), feasibility (``ok`` = structure integrator
returned a solution), and the Phase-2 dMdt>0 abort flag (``dmdt_ok``) -- plus
the solver's accepted root (``accept``).

Convergence metric (pure solver): total_residual = f_E**2 + f_T**2, converged
when < 1e-4 (trinity.phase1b_energy_implicit.get_betadelta.RESIDUAL_THRESHOLD).
Legacy clamp box = beta in [0, 1], delta in [-1, 0] (BETA/DELTA_MIN/MAX); the
scan deliberately reaches outside it.

Per config the segment shown is the one whose accepted root has the lowest
residual (the closest each config gets to convergence). Produces:
  - betadelta_residual.png : 7x7 residual landscape + root + clamp box (2x2)
  - betadelta_regime.png   : feasibility/abort regime map + root + box (2x2)

Usage: python scratch/phase2/analyze_probe.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
RES_THRESH = 1e-4  # RESIDUAL_THRESHOLD
BETA_BOX, DELTA_BOX = (0.0, 1.0), (-1.0, 0.0)  # legacy clamp
BETA_AX = np.linspace(-1.0, 2.0, 7)  # SCAN_BETA from probe.py
DELTA_AX = np.linspace(-1.0, 0.5, 7)  # SCAN_DELTA

# (file, short label) -- segment auto-picked as the lowest-residual accept
CONFIGS = [
    ("probe_cloud1e6.jsonl", "flat   1e6, n=1e5, α=0"),
    ("probe_cloudPL.jsonl", "steep  1e6, α=−2"),
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
    """Segment whose accepted root has the smallest total_residual."""
    accepts = [r for r in recs if r["kind"] == "accept"]
    best = min(accepts, key=lambda r: r["total_residual"])
    return best["segment"], best


def grids(recs, seg):
    """Build 7x7 residual + regime grids for one segment's scan."""
    res = np.full((7, 7), np.nan)  # [delta_idx, beta_idx]
    regime = np.full((7, 7), np.nan)  # 0 no-structure, 1 dMdt<=0, 2 feasible
    for r in recs:
        if r["kind"] != "scan" or r["segment"] != seg:
            continue
        i = int(np.argmin(np.abs(BETA_AX - r["beta"])))
        j = int(np.argmin(np.abs(DELTA_AX - r["delta"])))
        if r.get("ok"):
            res[j, i] = r["f_E"] ** 2 + r["f_T"] ** 2
            regime[j, i] = 2.0 if r.get("dmdt_ok") else 1.0
        else:
            regime[j, i] = 0.0
    return res, regime


def _clampbox(ax):
    ax.add_patch(
        Rectangle(
            (BETA_BOX[0], DELTA_BOX[0]),
            BETA_BOX[1] - BETA_BOX[0],
            DELTA_BOX[1] - DELTA_BOX[0],
            fill=False,
            edgecolor="cyan",
            lw=2.0,
            ls="-",
            zorder=5,
        )
    )


def _root(ax, acc):
    conv = acc["converged"]
    ax.plot(
        acc["beta"],
        acc["delta"],
        marker="*",
        ms=20,
        mfc="lime" if conv else "red",
        mec="k",
        mew=1.0,
        ls="none",
        zorder=6,
    )


def _frame(ax, label, acc):
    ax.set_xlim(-1.3, 2.3)
    ax.set_ylim(-1.2, 0.7)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\delta$")
    tag = "converged" if acc["converged"] else f"res={acc['total_residual']:.1e}"
    ax.set_title(f"{label}\nseg {acc['segment']}, t={acc['t_now']:.3g} Myr · {tag}", fontsize=9)


def plot_residual(data, path):
    fin = [d[1][np.isfinite(d[1])] for d in data]
    allres = np.concatenate([a for a in fin if a.size])
    norm = LogNorm(vmin=max(allres.min(), 1e-6), vmax=allres.max())
    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad("0.8")  # no-structure cells

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), constrained_layout=True)
    for ax, (label, res, _regime, acc) in zip(axes.flat, data):
        m = ax.pcolormesh(
            BETA_AX, DELTA_AX, np.ma.masked_invalid(res), norm=norm, cmap=cmap, shading="nearest"
        )
        if np.nanmin(res) < RES_THRESH < np.nanmax(res):
            ax.contour(BETA_AX, DELTA_AX, res, levels=[RES_THRESH], colors="white", linewidths=1.4)
        _clampbox(ax)
        _root(ax, acc)
        _frame(ax, label, acc)
    cb = fig.colorbar(m, ax=axes, shrink=0.85, label=r"residual  $f_E^2+f_T^2$  (log)")
    cb.ax.axhline(RES_THRESH, color="white", lw=1.4)
    handles = [
        plt.Line2D([], [], marker="*", mfc="lime", mec="k", ls="none", ms=13, label="root (conv)"),
        plt.Line2D(
            [], [], marker="*", mfc="red", mec="k", ls="none", ms=13, label="root (no conv)"
        ),
        Patch(fill=False, edgecolor="cyan", lw=2, label="legacy clamp box"),
        Patch(facecolor="0.8", label="no structure"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.02)
    )
    fig.suptitle(
        r"(β, δ) residual landscape — root vs the legacy clamp box  (conv threshold "
        f"{RES_THRESH:g} marked on bar)",
        fontsize=12,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_regime(data, path):
    cmap = ListedColormap(["#d62728", "#ff7f0e", "#2ca02c"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), constrained_layout=True)
    for ax, (label, _res, regime, acc) in zip(axes.flat, data):
        ax.pcolormesh(
            BETA_AX, DELTA_AX, np.ma.masked_invalid(regime), norm=norm, cmap=cmap, shading="nearest"
        )
        _clampbox(ax)
        _root(ax, acc)
        _frame(ax, label, acc)
    handles = [
        Patch(facecolor="#2ca02c", label="feasible (dMdt>0)"),
        Patch(facecolor="#ff7f0e", label="structure, dMdt≤0 (abort)"),
        Patch(facecolor="#d62728", label="no structure / timeout"),
        Patch(fill=False, edgecolor="cyan", lw=2, label="legacy clamp box"),
        plt.Line2D(
            [], [], marker="*", mfc="lime", mec="k", ls="none", ms=13, label="accepted root"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "(β, δ) feasibility regime — where a valid bubble structure exists vs the clamp box",
        fontsize=12,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fg_grids(recs, seg):
    """7x7 f-metric and g-metric residual grids for one segment's scan.

    f_total = f_E**2 + f_T**2 (legacy; f_E denominator Edot_from_beta -> pole).
    g_total = g_E**2 + f_T**2 with g_E = (E1 - E2)/Lmech_total (pole-free).
    E1=Edot_from_beta, E2=Edot_from_balance, Lg=L_gain=Lmech_total (recorded).
    Cells with no bubble structure stay NaN (masked).
    """
    fgrid = np.full((7, 7), np.nan)
    ggrid = np.full((7, 7), np.nan)
    for r in recs:
        if r["kind"] != "scan" or r["segment"] != seg or not r.get("ok"):
            continue
        i = int(np.argmin(np.abs(BETA_AX - r["beta"])))
        j = int(np.argmin(np.abs(DELTA_AX - r["delta"])))
        fgrid[j, i] = r["f_E"] ** 2 + r["f_T"] ** 2
        g_E = (r["E1"] - r["E2"]) / r["Lg"]
        ggrid[j, i] = g_E**2 + r["f_T"] ** 2
    return fgrid, ggrid


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
            ax.set_xlim(-1.3, 2.3)
            ax.set_ylim(-1.2, 0.7)
            ax.set_xlabel(r"$\beta$")
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
        "Same scan, two metrics — the f-pole (dark stripe at β≈1.5) vanishes under g",
        fontsize=12,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    data = []
    for fn, label in CONFIGS:
        recs = load(fn)
        seg, acc = pick_segment(recs)
        res, regime = grids(recs, seg)
        data.append((label, res, regime, acc))
    plot_residual(data, HERE / "betadelta_residual.png")
    plot_regime(data, HERE / "betadelta_regime.png")

    # f-vs-g comparison for a converger (flat) and a staller (steep)
    fg = []
    for fn, label in (CONFIGS[0], CONFIGS[1]):
        recs = load(fn)
        seg, acc = pick_segment(recs)
        fgrid, ggrid = fg_grids(recs, seg)
        fg.append((label, fgrid, ggrid, acc))
    plot_f_vs_g(fg, HERE / "betadelta_f_vs_g.png")
    print("wrote betadelta_residual.png, betadelta_regime.png, betadelta_f_vs_g.png")


if __name__ == "__main__":
    main()
