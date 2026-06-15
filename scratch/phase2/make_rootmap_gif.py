#!/usr/bin/env python3
"""Animated cage-vs-no-cage root-finding GIF for the steep run -- a PURE READ.

All the expensive physics (the real legacy/caged solve and the re-solved interior
profiles) is tabulated once by tabulate_cage.py into two COMMITTED csvs
(rootmap_cage_scalars.csv per segment + rootmap_cage_profiles.csv long-format);
this script only reads those and renders, so it runs in seconds, is cheap to
re-style, and reproduces from git after the container is gone. Run
tabulate_cage.py first (needs the pinned venv); this script needs only
numpy + pandas + matplotlib + pillow.

Frame = one energy-implicit segment (increasing time). Five panels:
  LEFT  A : the (beta,delta) plane. Cyan box = the legacy clamp ("the cage").
            hybr roots (no cage, circles) escape the box; the REAL legacy/caged
            roots (squares) -- the actual bounded solve, not a geometric clip --
            ride the edge. Both accumulate, coloured by time; a dashed connector
            marks the per-segment clamp error.
  LEFT  B : residual g of the two arms vs t (g<1e-4 = converged). hybr converges;
            the cage cannot (it is structurally forbidden the out-of-box root).
  LEFT  C : interior density n(r) [cm^-3] vs radial fraction, cage vs no cage.
  RIGHT D : interior velocity v(r) [pc/Myr] vs radial fraction, cage vs no cage
            (inflow = v<0; the cage's monotone solve hides the surge inflow).
  RIGHT E : Lmech_W / Lmech_SN / Lmech_total vs t, marker at the current t.

  python scratch/phase2/make_rootmap_gif.py
Writes rootmap_cage.gif.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
SCALARS = HERE / "rootmap_cage_scalars.csv"
PROFILES = HERE / "rootmap_cage_profiles.csv.gz"
BOX_B, BOX_D = (0.0, 1.0), (-1.0, 0.0)  # the cage (legacy clamp)
THRESH = 1e-4
FPS = 9
HYBR_C, CAGE_C = "#0072B2", "0.45"  # no-cage (blue) vs cage (grey)

_STYLE = HERE.parents[1] / "paper" / "_lib" / "trinity.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))
plt.rcParams["text.usetex"] = False


def load():
    if not SCALARS.exists() or not PROFILES.exists():
        raise SystemExit(
            f"missing {SCALARS.name}/{PROFILES.name} -- run "
            f"`python {HERE.name}/tabulate_cage.py` first"
        )
    scal = pd.read_csv(SCALARS)
    prof = pd.read_csv(PROFILES)
    nseg = len(scal)
    nf = int((prof["segment"] == 0).sum())
    f_grid = prof["f"].to_numpy()[:nf]
    rs = lambda c: prof[c].to_numpy().reshape(nseg, nf)  # noqa: E731
    return dict(
        t=scal["t"].to_numpy(),
        hb=scal["beta_nocage"].to_numpy(),
        hd=scal["delta_nocage"].to_numpy(),
        cb=scal["beta_cage"].to_numpy(),
        cd=scal["delta_cage"].to_numpy(),
        g_h=scal["g_nocage"].to_numpy(),
        g_c=scal["g_cage"].to_numpy(),
        cage_ok=scal["cage_ok"].to_numpy().astype(bool),
        R2=scal["R2"].to_numpy(),
        lw=scal["Lmech_W"].to_numpy() / 1e8,
        lsn=scal["Lmech_SN"].to_numpy() / 1e8,
        lt=scal["Lmech_total"].to_numpy() / 1e8,
        f_grid=f_grid,
        v_h=rs("v_nocage"),
        n_h=rs("n_nocage_cm3"),
        v_c=rs("v_cage"),
        n_c=rs("n_cage_cm3"),
    )


def main():
    d = load()
    t, hb, hd, cb, cd = d["t"], d["hb"], d["hd"], d["cb"], d["cd"]
    g_h, g_c, cage_ok = d["g_h"], d["g_c"], d["cage_ok"]
    R2 = d["R2"]
    lw, lsn, lt = d["lw"], d["lsn"], d["lt"]
    f_grid, v_h, n_h, v_c, n_c = d["f_grid"], d["v_h"], d["n_h"], d["v_c"], d["n_c"]
    n = len(t)
    tnorm = Normalize(t.min(), t.max())

    # static axis limits (so nothing jumps frame-to-frame)
    allb = np.concatenate([hb, cb[cage_ok]])
    alld = np.concatenate([hd, cd[cage_ok]])
    xlo, xhi = min(allb.min(), BOX_B[0]), max(allb.max(), BOX_B[1])
    ylo, yhi = min(alld.min(), BOX_D[0]), max(alld.max(), BOX_D[1])
    xpad, ypad = 0.08 * (xhi - xlo) + 0.05, 0.08 * (yhi - ylo) + 0.05
    XLIM, YLIM = (xlo - xpad, xhi + xpad), (ylo - ypad, yhi + ypad)
    vlo = float(np.nanmin([np.nanmin(v_h), np.nanmin(v_c)]))
    vhi = float(np.nanmax([np.nanmax(v_h), np.nanmax(v_c)]))
    VLIM = (min(vlo, -0.2) - 0.05 * abs(vhi), vhi * 1.05)
    nlo = float(np.nanmin([np.nanmin(n_h[n_h > 0]), np.nanmin(n_c[n_c > 0])]))
    nhi = float(np.nanmax([np.nanmax(n_h), np.nanmax(n_c)]))

    fig, axd = plt.subplot_mosaic(
        [["A", "D"], ["A", "D"], ["B", "D"], ["C", "E"]],
        figsize=(14, 9.5),
        width_ratios=[1.5, 1.0],
        height_ratios=[1.15, 1.15, 0.95, 0.95],
        constrained_layout=True,
    )
    aA, aB, aC, aD, aE = (axd[k] for k in "ABCDE")

    def update(i):
        out = not (BOX_B[0] <= hb[i] <= BOX_B[1] and BOX_D[0] <= hd[i] <= BOX_D[1])
        m = slice(0, i + 1)
        ok = cage_ok[: i + 1]

        # ---- A: (beta, delta) plane, cage vs no cage ----
        aA.clear()
        aA.add_patch(
            Rectangle((BOX_B[0], BOX_D[0]), 1, 1, fill=False, edgecolor="cyan", lw=2.5, zorder=2)
        )
        aA.text(
            0.5,
            -0.5,
            "the cage\n(legacy box)",
            color="darkcyan",
            ha="center",
            va="center",
            fontsize=8,
            zorder=2,
        )
        aA.plot(hb, hd, color="0.85", lw=1.0, zorder=1)  # full hybr path (context)
        for k in range(i + 1):  # connectors cage -> hybr (clamp error)
            if cage_ok[k]:
                aA.plot([cb[k], hb[k]], [cd[k], hd[k]], color="0.7", lw=0.6, alpha=0.5, zorder=1)
        aA.scatter(
            cb[m][ok],
            cd[m][ok],
            c=t[m][ok],
            cmap="viridis",
            norm=tnorm,
            marker="s",
            s=20,
            edgecolor="0.3",
            lw=0.3,
            zorder=3,
            label="cage (real legacy solve)",
        )
        aA.scatter(
            hb[m],
            hd[m],
            c=t[m],
            cmap="viridis",
            norm=tnorm,
            s=26,
            edgecolor="k",
            lw=0.3,
            zorder=4,
            label="no cage (hybr)",
        )
        if out and cage_ok[i]:
            aA.plot([cb[i], hb[i]], [cd[i], hd[i]], color="crimson", lw=1.4, ls="--", zorder=5)
        if cage_ok[i]:
            aA.plot(
                cb[i],
                cd[i],
                marker="s",
                ms=12,
                mfc="crimson",
                mec="k",
                mew=1.2,
                ls="none",
                zorder=6,
            )
        aA.plot(
            hb[i], hd[i], marker="*", ms=22, mfc="#ffd000", mec="k", mew=1.2, ls="none", zorder=7
        )
        aA.set_xlim(*XLIM)
        aA.set_ylim(*YLIM)
        aA.set_xlabel(r"$\beta$")
        aA.set_ylabel(r"$\delta$")
        tag = "OUTSIDE the cage" if out else "inside the cage"
        aA.set_title(
            f"Root finding with vs without the cage  (t={t[i]:.2f} Myr; {tag})\n"
            f"hybr  β={hb[i]:+.2f}, δ={hd[i]:+.2f}   |   cage  β={cb[i]:+.2f}, δ={cd[i]:+.2f}",
            fontsize=10,
        )
        aA.text(
            0.02,
            0.98,
            "marker colour = time →",
            transform=aA.transAxes,
            fontsize=8,
            va="top",
            color="0.45",
        )
        aA.legend(loc="lower right", fontsize=8)
        aA.grid(alpha=0.3)

        # ---- B: residual g of the two arms vs t ----
        aB.clear()
        aB.plot(
            t[m], np.clip(g_h[m], 1e-7, None), "-o", ms=2.5, color=HYBR_C, label="no cage (hybr)"
        )
        aB.plot(
            t[m], np.clip(g_c[m], 1e-7, None), "-s", ms=2.5, color="crimson", label="cage (legacy)"
        )
        aB.axhline(THRESH, color="k", ls="--", lw=1, label=f"converged < {THRESH:g}")
        aB.set_yscale("log")
        aB.set_ylim(4e-7, 1e2)
        aB.set_xlim(t.min(), t.max())
        aB.set_xlabel("t  [Myr]")
        aB.set_ylabel(r"residual $g$")
        aB.set_title("convergence — two arms", fontsize=9)
        aB.legend(fontsize=7, loc="upper left", ncol=1)
        aB.grid(alpha=0.3)

        # ---- C: interior density vs radius ----
        aC.clear()
        aC.semilogy(f_grid, n_h[i], color=HYBR_C, lw=1.6, label="no cage")
        if cage_ok[i]:
            aC.semilogy(f_grid, n_c[i], color=CAGE_C, lw=1.6, ls="--", label="cage")
        aC.set_xlim(0, 1)
        aC.set_ylim(nlo * 0.7, nhi * 1.4)
        aC.set_xlabel("radial fraction  (0 = R1, 1 = R2)")
        aC.set_ylabel(r"$n(r)$  [cm$^{-3}$]")
        aC.set_title("interior density", fontsize=9)
        aC.legend(fontsize=7, loc="upper left")
        aC.grid(alpha=0.3, which="both")

        # ---- D: interior velocity vs radius ----
        aD.clear()
        aD.axhline(0.0, color="k", lw=0.8)
        aD.axhspan(VLIM[0], 0.0, color="r", alpha=0.06)
        aD.plot(
            f_grid, v_h[i], color=HYBR_C, lw=1.9, label=f"no cage  (v_min={np.nanmin(v_h[i]):+.2f})"
        )
        if cage_ok[i]:
            aD.plot(
                f_grid,
                v_c[i],
                color=CAGE_C,
                lw=1.9,
                ls="--",
                label=f"cage  (v_min={np.nanmin(v_c[i]):+.2f})",
            )
        aD.set_xlim(0, 1)
        aD.set_ylim(*VLIM)
        aD.set_xlabel("radial fraction  (0 = R1, 1 = R2)")
        aD.set_ylabel("v(r)  [pc/Myr]")
        inflow = float(np.nanmin(v_h[i])) < -0.01
        aD.set_title(
            f"interior velocity  (R2={R2[i]:.2f} pc)" + ("  — INFLOW (v<0)" if inflow else ""),
            fontsize=10,
            color="#b30000" if inflow else "k",
        )
        aD.legend(fontsize=8, loc="upper left")
        aD.grid(alpha=0.3)

        # ---- E: Lmech vs t ----
        aE.clear()
        aE.plot(t, lt, color="k", lw=1.6, ls="--", label=r"$L_{\rm tot}$")
        aE.plot(t, lw, color=HYBR_C, lw=1.4, label=r"$L_{\rm W}$ (wind)")
        aE.plot(t, lsn, color="#9467bd", lw=1.4, label=r"$L_{\rm SN}$")
        aE.axvline(t[i], color="crimson", lw=1.4)
        aE.set_xlim(t.min(), t.max())
        aE.set_xlabel("t  [Myr]")
        aE.set_ylabel(r"$L_{\rm mech}$ [$10^8$]")
        aE.set_title("feedback power", fontsize=9)
        aE.legend(fontsize=7, loc="upper left", ncol=3)
        aE.grid(alpha=0.3)
        return []

    anim = FuncAnimation(fig, update, frames=n, interval=1000 / FPS, blit=False)
    out = HERE / "rootmap_cage.gif"
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=85)
    plt.close(fig)
    print(f"wrote {out}  ({n} frames)")


if __name__ == "__main__":
    main()
