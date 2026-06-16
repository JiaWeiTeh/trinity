#!/usr/bin/env python3
"""Animated arms_rootmap: the cage vs no cage, straight from arms_*.jsonl.

This is arms_rootmap.png but revealed over segments (= time) instead of all at
once. Everything is tabulated in the jsonl (per-segment beta, delta, g for each
arm), so it is a pure read -- NO structure re-solve -- and renders in seconds.

  arm A = production (clamped to the legacy box)  -> "the cage"
  arm D = hybr (unbounded)                         -> "no cage"

Two panels per frame (frame = energy-implicit segment):
  LEFT  : (beta, delta) plane + the legacy box. A (clamped, squares) and D (hybr,
          circles) accumulate over time; a connector marks the per-segment clamp
          error. D leaves the box; A rides its edge.
  RIGHT : residual g at the accepted root vs t, per arm (g<1e-4 = converged).

Needs pillow (matplotlib PillowWriter). No trinity / no venv required.
  python docs/dev/scratch/phase2/make_arms_rootmap_gif.py [config]   # default arms_simple1e5
Writes arms_rootmap_<config>.gif.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
BOX_B, BOX_D = (0.0, 1.0), (-1.0, 0.0)  # legacy clamp = "the cage"
THRESH = 1e-4
FPS = 6

_STYLE = HERE.parents[1] / "paper" / "_lib" / "trinity.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))
plt.rcParams["text.usetex"] = False


def load(cfg):
    by = defaultdict(dict)  # by[arm][segment] = record
    for line in open(HERE / f"{cfg}.jsonl"):
        if line.strip():
            r = json.loads(line)
            by[r["arm"]][r["segment"]] = r
    return by


def main():
    cfg = sys.argv[1] if len(sys.argv) > 1 else "arms_simple1e5"
    by = load(cfg)
    segs = sorted(set(by["A"]) | set(by["D"]))
    A = [by["A"].get(s) for s in segs]  # production (cage); always present
    D = [by["D"].get(s) for s in segs]  # hybr (no cage); some aborts (no beta)

    def xy(rec):
        if rec is None or "beta" not in rec:
            return None, None
        return rec["beta"], rec["delta"]

    ax_a = [xy(r) for r in A]
    ax_d = [xy(r) for r in D]
    tA = np.array([r["t_now"] if r else np.nan for r in A])
    gA = np.array([r.get("g", np.nan) if r else np.nan for r in A])
    tD = np.array([r["t_now"] if r else np.nan for r in D])
    gD = np.array([(r.get("g", np.nan) if (r and "g" in r) else np.nan) for r in D])
    snorm = Normalize(segs[0], segs[-1])

    # axis limits from the actual data (D leaves the box) + the box, with padding
    allb = [b for (b, d) in ax_a + ax_d if b is not None]
    alld = [d for (b, d) in ax_a + ax_d if d is not None]
    xlo, xhi = min(allb + [BOX_B[0]]), max(allb + [BOX_B[1]])
    ylo, yhi = min(alld + [BOX_D[0]]), max(alld + [BOX_D[1]])
    xpad, ypad = 0.08 * (xhi - xlo) + 0.05, 0.08 * (yhi - ylo) + 0.05
    XLIM, YLIM = (xlo - xpad, xhi + xpad), (ylo - ypad, yhi + ypad)

    fig, (aL, aR) = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    def update(i):
        aL.clear()
        aL.add_patch(Rectangle((BOX_B[0], BOX_D[0]), 1, 1, fill=False, edgecolor="cyan", lw=2.5))
        aL.text(0.5, -0.5, "the cage\n(legacy box)", color="darkcyan", ha="center", fontsize=8)
        for k in range(i + 1):  # connectors A -> D (clamp error)
            (ba, da), (bd, dd) = ax_a[k], ax_d[k]
            if ba is not None and bd is not None:
                aL.plot([ba, bd], [da, dd], color="0.7", lw=0.7, alpha=0.6, zorder=1)
        sa = [k for k in range(i + 1) if ax_a[k][0] is not None]
        sd = [k for k in range(i + 1) if ax_d[k][0] is not None]
        if sa:
            aL.scatter(
                [ax_a[k][0] for k in sa],
                [ax_a[k][1] for k in sa],
                c=[segs[k] for k in sa],
                cmap="viridis",
                norm=snorm,
                marker="s",
                s=26,
                edgecolor="0.3",
                lw=0.3,
                zorder=2,
                label="A production (cage)",
            )
        if sd:
            aL.scatter(
                [ax_d[k][0] for k in sd],
                [ax_d[k][1] for k in sd],
                c=[segs[k] for k in sd],
                cmap="viridis",
                norm=snorm,
                s=40,
                edgecolor="k",
                lw=0.4,
                zorder=3,
                label="D hybr (no cage)",
            )
        for cur, mk, ms in ((ax_a[i], "s", 13), (ax_d[i], "*", 20)):  # highlight current
            if cur[0] is not None:
                aL.plot(
                    cur[0],
                    cur[1],
                    marker=mk,
                    ms=ms,
                    mfc="#ffd000",
                    mec="k",
                    mew=1.2,
                    ls="none",
                    zorder=5,
                )
        aL.set_xlim(*XLIM)
        aL.set_ylim(*YLIM)
        aL.set_xlabel(r"$\beta$")
        aL.set_ylabel(r"$\delta$")
        aL.set_title(
            f"{cfg.replace('arms_', '')}: cage (A) vs no cage (D)   "
            f"(segment {segs[i]}, t={tD[i] if not np.isnan(tD[i]) else tA[i]:.3g} Myr)",
            fontsize=10,
        )
        aL.legend(fontsize=8, loc="lower right")
        aL.grid(alpha=0.3)

        aR.clear()
        m = slice(0, i + 1)
        aR.plot(tA[m], np.clip(gA[m], 1e-7, None), "-s", ms=3, color="0.5", label="A (cage)")
        aR.plot(tD[m], np.clip(gD[m], 1e-7, None), "-o", ms=3, color="#2ca02c", label="D (hybr)")
        aR.axhline(THRESH, color="k", ls="--", lw=1, label=f"threshold {THRESH:g}")
        aR.set_yscale("log")
        aR.set_ylim(4e-7, 1e2)
        aR.set_xlim(np.nanmin(tA), np.nanmax(tA))
        aR.set_xlabel("t  [Myr]")
        aR.set_ylabel(r"residual $g$ at accepted root")
        aR.set_title(
            "convergence — residual g at the accepted root (g<1e-4 = converged)", fontsize=10
        )
        aR.legend(fontsize=8, loc="lower left")
        aR.grid(alpha=0.3)
        return []

    anim = FuncAnimation(fig, update, frames=len(segs), interval=1000 / FPS, blit=False)
    out = HERE / f"arms_rootmap_{cfg.replace('arms_', '')}.gif"
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=95)
    plt.close(fig)
    print(f"wrote {out}  ({len(segs)} frames)")


if __name__ == "__main__":
    main()
