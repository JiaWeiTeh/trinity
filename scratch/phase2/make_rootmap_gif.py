#!/usr/bin/env python3
"""Animated cage-vs-no-cage root-finding GIF for the steep run -- a PURE READ.

All the expensive physics (the real legacy/caged solve and the re-solved interior
profiles) is tabulated once by tabulate_cage.py into two COMMITTED csvs
(rootmap_cage_scalars.csv per segment + rootmap_cage_profiles.csv.gz long-format);
this script only reads those and renders, so it runs in seconds, is cheap to
re-style, and reproduces from git after the container is gone. Run
tabulate_cage.py first (needs the pinned venv); this script needs only
numpy + pandas + matplotlib + pillow.

Frame pacing: the run samples t very unevenly (dt ramps 3e-4 -> capped 5e-2, so
~35% of segments live in the first 3% of physical time). Playing one frame per
segment would dwell on the early phase. Frames are laid on a uniform grid in t
(linear): the late phase is already uniform (dt capped) so the grid is DENSER than
the segments there -- preserving sharp 1-segment features like the surge inflow --
while the over-dense early phase is decimated, fixing the dwell. (log-t was tried
first but under-samples the late phase and washes out the inflow spike.) The moving
star/square and the profile curves are INTERPOLATED onto the grid for smooth
motion; the accumulated (beta,delta) scatter and the residual history stay at the
REAL segment times (dots), so nothing fabricated hides the true samples.

Six panels, current time = the frame's t grid point:
  LEFT  A : the (beta,delta) plane. Cyan box = the legacy clamp ("the cage").
            hybr roots (no cage) escape the box; the REAL legacy/caged roots
            (squares) -- the actual bounded solve, not a geometric clip -- ride
            the edge. Dots = real segments (colour = time); a dashed connector
            marks the current clamp error.
  LEFT  B : residual g of the two arms vs t (g<1e-4 = converged). hybr converges;
            the cage cannot (it is structurally forbidden the out-of-box root).
  LEFT  C : interior velocity v(r) vs PHYSICAL radius r [pc] (R1->R2), cage vs no
            cage -- shows the bubble physically growing as the axis fills.
  RIGHT D : interior velocity v(r) vs radial fraction (0=R1, 1=R2), cage vs no cage
            (inflow = v<0; the cage's monotone solve hides the surge inflow).
  RIGHT E : Lmech_W / Lmech_SN / Lmech_total vs t, marker at the current t.
  RIGHT F : bubble radius R2 and ionization front R_IF vs t (R_IF~R2 for a dense
            shell; they diverge for a less-dense / optically-thin shell).

  python scratch/phase2/make_rootmap_gif.py
Writes rootmap_cage.gif.
"""

import argparse
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
DEFAULT_PREFIX = "rootmap_cage"  # steep run; --prefix selects another tabulation
BOX_B, BOX_D = (0.0, 1.0), (-1.0, 0.0)  # the cage (legacy clamp)
THRESH = 1e-4
NFRAMES = 150  # frames on the uniform (linear) t grid
FPS = 12
HYBR_C, CAGE_C = "#0072B2", "0.45"  # no-cage (blue) vs cage (grey)

_STYLE = HERE.parents[1] / "paper" / "_lib" / "trinity.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))
plt.rcParams["text.usetex"] = False


def load(scalars, profiles):
    if not scalars.exists() or not profiles.exists():
        raise SystemExit(
            f"missing {scalars.name}/{profiles.name} -- run "
            f"`python {HERE.name}/tabulate_cage.py` first"
        )
    scal = pd.read_csv(scalars)
    prof = pd.read_csv(profiles)
    nseg = len(scal)
    nf = int((prof["segment"] == 0).sum())
    f_grid = prof["f"].to_numpy()[:nf]
    rs = lambda c: prof[c].to_numpy().reshape(nseg, nf)  # noqa: E731
    col = lambda c: scal[c].to_numpy() if c in scal else np.full(nseg, np.nan)  # noqa: E731
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
        R1_h=scal["R1_nocage"].to_numpy(),
        R1_c=scal["R1_cage"].to_numpy(),
        R_IF=col("R_IF_nocage"),
        lw=scal["Lmech_W"].to_numpy() / 1e8,
        lsn=scal["Lmech_SN"].to_numpy() / 1e8,
        lt=scal["Lmech_total"].to_numpy() / 1e8,
        f_grid=f_grid,
        v_h=rs("v_nocage"),
        v_c=rs("v_cage"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help="CSV/gif basename prefix")
    ap.add_argument("--label", default="", help="optional config label shown in the title")
    args = ap.parse_args()
    lbl = f" — {args.label}" if args.label else ""
    d = load(HERE / f"{args.prefix}_scalars.csv", HERE / f"{args.prefix}_profiles.csv.gz")
    t = d["t"]
    hb, hd, cb, cd = d["hb"], d["hd"], d["cb"], d["cd"]
    g_h, g_c, cage_ok = d["g_h"], d["g_c"], d["cage_ok"]
    R2, R1_h, R1_c, R_IF = d["R2"], d["R1_h"], d["R1_c"], d["R_IF"]
    lw, lsn, lt = d["lw"], d["lsn"], d["lt"]
    f_grid, v_h, v_c = d["f_grid"], d["v_h"], d["v_c"]
    tnorm = Normalize(t.min(), t.max())

    # ---- frame grid: uniform in t; denser than the (capped-dt) late segments so the
    # surge inflow survives, decimating the over-dense early phase ----
    tg = np.linspace(t[0], t[-1], NFRAMES)
    lin = lambda y: np.interp(tg, t, y)  # noqa: E731  smooth current-marker value
    logi = lambda y: 10 ** np.interp(tg, t, np.log10(np.clip(y, 1e-7, None)))  # noqa: E731
    hb_f, hd_f, cb_f, cd_f = lin(hb), lin(hd), lin(cb), lin(cd)
    R2_f, R1h_f, R1c_f, RIF_f = lin(R2), lin(R1_h), lin(R1_c), lin(R_IF)
    gh_f, gc_f = logi(g_h), logi(g_c)
    cage_f = lin(cage_ok.astype(float)) > 0.999  # cage curves valid at this frame?

    def prof_interp(P):  # interpolate each radial-fraction column onto tg
        return np.column_stack([np.interp(tg, t, P[:, j]) for j in range(P.shape[1])])

    vh_f, vc_f = prof_interp(v_h), prof_interp(v_c)

    # ---- static axis limits (from the real data, so nothing jumps) ----
    allb = np.concatenate([hb, cb[cage_ok]])
    alld = np.concatenate([hd, cd[cage_ok]])
    xlo, xhi = min(allb.min(), BOX_B[0]), max(allb.max(), BOX_B[1])
    ylo, yhi = min(alld.min(), BOX_D[0]), max(alld.max(), BOX_D[1])
    xpad, ypad = 0.08 * (xhi - xlo) + 0.05, 0.08 * (yhi - ylo) + 0.05
    XLIM, YLIM = (xlo - xpad, xhi + xpad), (ylo - ypad, yhi + ypad)
    vlo = float(np.nanmin([np.nanmin(v_h), np.nanmin(v_c)]))
    vhi = float(np.nanmax([np.nanmax(v_h), np.nanmax(v_c)]))
    VLIM = (min(vlo, -0.2) - 0.05 * abs(vhi), vhi * 1.05)
    RPC_MAX = float(np.nanmax(R2)) * 1.02

    fig, axd = plt.subplot_mosaic(
        [["A", "D"], ["A", "D"], ["B", "E"], ["C", "F"]],
        figsize=(14, 9.5),
        width_ratios=[1.5, 1.0],
        height_ratios=[1.2, 1.2, 0.9, 0.9],
        constrained_layout=True,
    )
    aA, aB, aC, aD, aE, aF = (axd[k] for k in "ABCDEF")

    def update(i):
        tc = tg[i]
        rm = t <= tc + 1e-12  # real segments revealed so far
        out = not (BOX_B[0] <= hb_f[i] <= BOX_B[1] and BOX_D[0] <= hd_f[i] <= BOX_D[1])

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
        rmc = rm & cage_ok
        aA.scatter(
            cb[rmc],
            cd[rmc],
            c=t[rmc],
            cmap="viridis",
            norm=tnorm,
            marker="s",
            s=18,
            edgecolor="0.3",
            lw=0.3,
            zorder=3,
            label="cage (real legacy solve)",
        )
        aA.scatter(
            hb[rm],
            hd[rm],
            c=t[rm],
            cmap="viridis",
            norm=tnorm,
            s=24,
            edgecolor="k",
            lw=0.3,
            zorder=4,
            label="no cage (hybr)",
        )
        if out and cage_f[i]:
            aA.plot(
                [cb_f[i], hb_f[i]], [cd_f[i], hd_f[i]], color="crimson", lw=1.4, ls="--", zorder=5
            )
        if cage_f[i]:
            aA.plot(
                cb_f[i],
                cd_f[i],
                marker="s",
                ms=12,
                mfc="crimson",
                mec="k",
                mew=1.2,
                ls="none",
                zorder=6,
            )
        aA.plot(
            hb_f[i],
            hd_f[i],
            marker="*",
            ms=22,
            mfc="#ffd000",
            mec="k",
            mew=1.2,
            ls="none",
            zorder=7,
        )
        aA.set_xlim(*XLIM)
        aA.set_ylim(*YLIM)
        aA.set_xlabel(r"$\beta$")
        aA.set_ylabel(r"$\delta$")
        tag = "OUTSIDE the cage" if out else "inside the cage"
        aA.set_title(
            f"Root finding with vs without the cage{lbl}  (t={tc:.3g} Myr; {tag})\n"
            f"hybr  β={hb_f[i]:+.2f}, δ={hd_f[i]:+.2f}   |   cage  β={cb_f[i]:+.2f}, δ={cd_f[i]:+.2f}",
            fontsize=10,
        )
        aA.text(
            0.02,
            0.98,
            "dots = real segments (colour = time)",
            transform=aA.transAxes,
            fontsize=8,
            va="top",
            color="0.45",
        )
        aA.legend(loc="lower right", fontsize=8)
        aA.grid(alpha=0.3)

        # ---- B: residual g of the two arms vs t (real history + current marker) ----
        aB.clear()
        aB.plot(
            t[rm], np.clip(g_h[rm], 1e-7, None), "-o", ms=2.5, color=HYBR_C, label="no cage (hybr)"
        )
        aB.plot(
            t[rm],
            np.clip(g_c[rm], 1e-7, None),
            "-s",
            ms=2.5,
            color="crimson",
            label="cage (legacy)",
        )
        aB.plot(tc, gh_f[i], "o", ms=8, mfc="#ffd000", mec="k", zorder=5)
        aB.plot(tc, gc_f[i], "s", ms=8, mfc="crimson", mec="k", zorder=5)
        aB.axhline(THRESH, color="k", ls="--", lw=1, label=f"converged < {THRESH:g}")
        aB.set_yscale("log")
        aB.set_ylim(4e-7, 1e2)
        aB.set_xlim(t.min(), t.max())
        aB.set_xlabel("t  [Myr]")
        aB.set_ylabel(r"residual $g$")
        aB.set_title("convergence — two arms", fontsize=9)
        aB.legend(fontsize=7, loc="upper left", ncol=1)
        aB.grid(alpha=0.3)

        # ---- C: interior velocity vs physical radius [pc] ----
        aC.clear()
        aC.axhline(0.0, color="k", lw=0.6)
        rpc_h = R1h_f[i] + f_grid * (R2_f[i] - R1h_f[i])
        aC.plot(rpc_h, vh_f[i], color=HYBR_C, lw=1.7, label="no cage")
        if cage_f[i]:
            rpc_c = R1c_f[i] + f_grid * (R2_f[i] - R1c_f[i])
            aC.plot(rpc_c, vc_f[i], color=CAGE_C, lw=1.7, ls="--", label="cage")
        aC.set_xlim(0, RPC_MAX)
        aC.set_ylim(*VLIM)
        aC.set_xlabel("radius  r  [pc]")
        aC.set_ylabel("v(r)  [pc/Myr]")
        aC.set_title("velocity vs physical radius", fontsize=9)
        aC.legend(fontsize=7, loc="upper left")
        aC.grid(alpha=0.3)

        # ---- D: interior velocity vs radial fraction ----
        aD.clear()
        aD.axhline(0.0, color="k", lw=0.8)
        aD.axhspan(VLIM[0], 0.0, color="r", alpha=0.06)
        aD.plot(
            f_grid,
            vh_f[i],
            color=HYBR_C,
            lw=1.9,
            label=f"no cage  (v_min={np.nanmin(vh_f[i]):+.2f})",
        )
        if cage_f[i]:
            aD.plot(
                f_grid,
                vc_f[i],
                color=CAGE_C,
                lw=1.9,
                ls="--",
                label=f"cage  (v_min={np.nanmin(vc_f[i]):+.2f})",
            )
        aD.set_xlim(0, 1)
        aD.set_ylim(*VLIM)
        aD.set_xlabel("radial fraction  (0 = R1, 1 = R2)")
        aD.set_ylabel("v(r)  [pc/Myr]")
        inflow = float(np.nanmin(vh_f[i])) < -0.01
        aD.set_title(
            f"interior velocity  (R2={R2_f[i]:.2f} pc)" + ("  — INFLOW (v<0)" if inflow else ""),
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
        aE.axvline(tc, color="crimson", lw=1.4)
        aE.set_xlim(t.min(), t.max())
        aE.set_xlabel("t  [Myr]")
        aE.set_ylabel(r"$L_{\rm mech}$ [$10^8$]")
        aE.set_title("feedback power", fontsize=9)
        aE.legend(fontsize=7, loc="upper left", ncol=3)
        aE.grid(alpha=0.3)

        # ---- F: bubble radius R2 & ionization front R_IF vs t ----
        aF.clear()
        aF.plot(t, R2, color=HYBR_C, lw=1.7, label=r"$R_2$ (bubble)")
        if np.isfinite(R_IF).any():
            aF.plot(t, R_IF, color="#d62728", lw=1.2, ls="--", label=r"$R_{\rm IF}$ (ion. front)")
        aF.axvline(tc, color="crimson", lw=1.0, alpha=0.6)
        aF.plot(tc, R2_f[i], "o", ms=7, mfc="#ffd000", mec="k", zorder=5)
        aF.set_xlim(t.min(), t.max())
        aF.set_xlabel("t  [Myr]")
        aF.set_ylabel("radius  [pc]")
        aF.set_title("bubble radius & ionization front", fontsize=9)
        aF.legend(fontsize=7, loc="upper left")
        aF.grid(alpha=0.3)
        return []

    anim = FuncAnimation(fig, update, frames=NFRAMES, interval=1000 / FPS, blit=False)
    out = HERE / f"{args.prefix}.gif"
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=85)
    plt.close(fig)
    print(f"wrote {out}  ({NFRAMES} frames, linear-t paced)")


if __name__ == "__main__":
    main()
